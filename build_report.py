"""Typeset the research report from verified, saved results using ReportLab."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results"
OUT = ROOT / "research_report.pdf"
W, H = A4
M, BODY = 46, W - 92
INK, MUTED, ACCENT, PALE = ("#173044", "#486170", "#006D65", "#C4D5D6")


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def csv_rows(name):
    with (DATA / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def setup_fonts():
    # The standard PDF fonts provide a portable fallback on a minimal Linux host.
    regular, bold, display = "Helvetica", "Helvetica-Bold", "Times-Bold"
    candidates = [("C:/Windows/Fonts", ("segoeui.ttf", "segoeuib.ttf", "georgia.ttf")),
                  ("/usr/share/fonts/truetype/dejavu",
                   ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSerif.ttf"))]
    for directory, files in candidates:
        paths = [Path(directory) / f for f in files]
        if all(p.exists() for p in paths):
            for label, p in zip(("Body", "BodyBold", "Display"), paths):
                pdfmetrics.registerFont(TTFont(label, str(p)))
            regular, bold, display = "Body", "BodyBold", "Display"
            break
    return regular, bold, display


class Report:
    def __init__(self):
        self.reg, self.bold, self.display = setup_fonts()
        self.c = Canvas(str(OUT), pagesize=A4, pageCompression=1)
        self.c.setTitle("How close can a quantum sampler get? Constrained portfolio selection")
        self.c.setAuthor("Quantum Portfolio Optimization research project")
        self.page = 0

    def label(self, x, y, text, size=9, font=None, color=INK):
        self.c.setFillColor(colors.HexColor(color))
        self.c.setFont(font or self.reg, size)
        self.c.drawString(x, H - y, text)

    def line(self, y, color=PALE):
        self.c.setStrokeColor(colors.HexColor(color))
        self.c.setLineWidth(.6)
        self.c.line(M, H - y, W - M, H - y)

    def para(self, y, text, size=11, leading=16, color=INK, width=BODY, x=M, font=None):
        style = ParagraphStyle("body", fontName=font or self.reg, fontSize=size,
                               leading=leading, textColor=colors.HexColor(color),
                               alignment=TA_LEFT, splitLongWords=False)
        para = Paragraph(text, style)
        _, h = para.wrap(width, H)
        para.drawOn(self.c, x, H - y - h)
        return y + h

    def head(self, y, text):
        self.label(M, y, text, 15, self.bold, ACCENT)
        return y + 8

    def start(self, section):
        if self.page:
            self.c.showPage()
        self.page += 1
        self.label(M, 31, "AMSTERDAM QISKIT FALL FEST     /     PORTFOLIO SELECTION", 8,
                   self.bold, ACCENT)
        self.line(42)
        self.label(M, H - 24, "RESEARCH REPORT     /     21 NOVEMBER 2026", 8, color=MUTED)
        self.label(W - 185, H - 24, f"{section.upper()}     /     {self.page:02d}", 8, color=MUTED)

    def title(self, title):
        self.label(M, 92, title, 27, self.display)

    def table(self, y, headers, rows, widths, height=26, size=10):
        if abs(sum(widths) - BODY) > .1:
            raise ValueError("Table column widths must span the text block")
        positions = [M]
        for width in widths:
            positions.append(positions[-1] + width)
        for j, name in enumerate(headers):
            self.label(positions[j] + 3, y, name, 8.4, self.bold, ACCENT)
        self.line(y + 7)
        for i, row in enumerate(rows):
            row_y = y + 7 + (i + 1) * height
            for j, value in enumerate(row):
                text = str(value)
                font = self.bold if j == 0 else self.reg
                if pdfmetrics.stringWidth(text, font, size) > widths[j] - 4:
                    raise ValueError(f"Table cell exceeds width: {text}")
                self.label(positions[j] + 3, row_y - 6, text, size, font)
            self.line(row_y + 1, "#E0E8E8")
        return y + 10 + len(rows) * height

    def figure(self, y, name, height):
        path = DATA / "figures" / name
        img = ImageReader(str(path))
        iw, ih = img.getSize()
        width = min(BODY, height * iw / ih)
        height = width * ih / iw
        self.c.drawImage(img, M + (BODY - width) / 2, H - y - height, width, height)
        return y + height

    def save(self):
        self.c.save()


def sized(value):
    return f"{int(value):,}" + ("*" if int(value) == 10000 else "")


def main():
    summary = csv_rows("by_size.csv")
    all_rows = csv_rows("instances.csv")
    off = [r for r in all_rows if r["scope"] == "official"]
    ext = summary[-1]
    qrows = load("quantum_audit.json")["instances"]
    repeats = load("repeats.json")["instances"]
    counter = load("counter_circuit.json")
    weighted = load("weighted_circuit_audit.json")
    max_gap = max(float(r["trained_gap_bp"]) for r in off)
    hits = sum(int(r["trained_hits"]) for r in summary[:-1])
    exact = sum(int(r["exact_references"]) for r in summary[:-1])
    report = Report()
    p = report.para
    h = report.head
    t = report.table
    report.start("Findings")
    report.label(M, 90, "How close can a quantum sampler get?", 26, report.display)
    y = p(110, "Risk, return and feasible portfolios across 35 benchmark universes and five "
          "10,000-asset stress tests", 16, 22, ACCENT)
    y = p(y + 17, "Choose exactly K assets. Stay under a price budget and a sector cap. "
          "Among the baskets that qualify, find the one with the best balance of expected "
          "return and shared risk. This study measures how much a simulated quantum sampler "
          "leaves on the table, using an independent classical optimizer and a global bound.")
    report.line(y + 19)
    report.label(M, y + 68, f"{max_gap:.3f}", 40, report.display, ACCENT)
    p(y + 34, "Largest observed loss against the exact optimum across the 35 official instances, "
      "in objective basis points. One objective bp equals 0.0001 in the challenge score.", 11.5,
      17, width=BODY-130, x=M+130)
    y += 99
    y = report.figure(y, "certified_gaps.png", 200)
    y = p(y + 7, "Figure 1. Each basket was checked against the organizer's original constraints. "
          "The vertical coordinate is the measured difference from the independently verified "
          "global optimum. The 10,000-asset group is separate.",
          9, 13, MUTED)
    y = h(y + 34, "The sampler matched 29 official optima")
    y = p(y + 4, f"The sampler matched a verified optimum on {hits} of {exact} official instances "
          "with exact references. At 10,000 assets its mean gap was "
          f"{float(ext['trained_gap_bp_mean']):.3f} objective bp. The classical solver was faster "
          "at every size on average. The large-state measurement law admits an efficient "
          "classical sampler, so these experiments establish solution quality on synthetic data.")
    h(y + 36, "A separate test of interference")
    y = p(y + 48, "A two-layer circuit over feasible portfolio addresses raised the mean single-shot "
          "chance of finding the exact optimum from 1.31% to 8.75% on 20 assets. The circuit "
          "uses an enumerated dictionary. Its costs and interpretation differ from the large-scale sampler.")

    report.start("Model")
    report.title("How a portfolio becomes an energy")
    y = p(113, "Each asset has an expected annual return, an individual variance and market, sector "
          "and style exposures. The generator supplies synthetic data for all three. Equal weights "
          "make a selected asset's portfolio share 1/K.")
    y = h(y + 32, "The score of one basket")
    y = p(y, "Risk = (squared length of the summed factor exposures + summed individual "
          "variances) / K². Return = average expected return of the K holdings.", 12, 18)
    y = p(y + 10, "Portfolio score = (risk − return) / 2. Lower is better.", 15, 21, ACCENT)
    y = p(y + 12, "For the 10-asset universe with seed 0, only 15 of the 45 pairs qualify. "
          "Assets A00001 and A00008 cost 80.55 against a budget of 111.97. Their expected "
          "return is 0.09673 and their variance is 0.06037, giving the exact minimum score "
          "−0.018179. Selecting highly correlated assets raises the squared-exposure term.")
    y = h(y + 32, "A quantum state with exactly K holdings")
    y = p(y, "Give each asset a qubit and a real parameter. For any K-asset basket, its "
          "measurement probability is proportional to the product of its asset weights. "
          "Doubling one asset's weight doubles the odds of every basket containing it "
          "relative to one that omits it. All other cardinalities have zero probability.")
    y = p(y + 8, "A suffix table counts the total weight of ways to fill the remaining slots. "
          "The ratio for taking the next asset sets a rotation angle. Following these "
          "conditional probabilities samples the same distribution as the coherent circuit. "
          "The initial preferences reward return and penalize marginal factor exposure.")
    y = h(y + 32, "Which measurements count")
    y = p(y, "An affordable basket within every sector cap receives the official score as its "
          "energy. Any violation adds more than the full attainable score span, plus a "
          "surcharge for the amount of excess. Eight rounds of 128 measurements adjust "
          "asset preferences and concentration. A final 256 shots brings each arm to 1,280. "
          "The reported basket is the best feasible raw measurement across those rounds.")
    y = p(y + 14, "The simulator uses n data bits and an O(nK) suffix table. A coherent "
          f"implementation adds a count register. A {counter['total_qubits']}-qubit "
          f"example compiled to {counter['all_to_all_counts']['cx']} CX gates at depth "
          f"{counter['all_to_all_depth']} with all-to-all connectivity. The 10,000-asset "
          "extension would require 10,010 logical wires in this construction. Only the "
          "conditional sampler was executed at that size.", 10.5, 15, MUTED)

    report.start("Benchmark")
    report.title("Quality and cost across the sweep")
    y = p(114, "Each row summarizes five independent generator seeds. Scores are shown in their "
          "original units. Gaps to the true optimum are in objective bp. Spread is the sample standard "
          "deviation across instances. All 40 returned baskets passed the official checks.")
    rows = []
    for d in summary:
        rows.append([sized(d["n"]), f"{float(d['trained_score_mean']):.5f} ± {float(d['trained_score_sd']):.5f}",
                     f"{float(d['trained_gap_bp_mean']):.3f} ± {float(d['trained_gap_bp_sd']):.3f}",
                     f"{float(d['classical_gap_bp_mean']):.3f}",
                     f"{d['trained_hits']}/{d['exact_references']}"])
    y = t(y+25, ["Assets", "Sampler score, mean ± SD", "Exact gap, mean ± SD",
                 "Classical gap¹", "Exact hits²"], rows, [61, 151, 139, 83, BODY-434], size=9.3)
    y = p(y + 8, "¹ Mean gap between the classical incumbent and the optimum in objective bp. "
          "² Sampler hits / available exact references, with agreement to 10⁻¹⁰ in score. "
          "* Separate extension. The full per-seed comparison and timing spreads are in the CSV tables.",
          9, 13, MUTED)
    y = h(y + 32, "Wall time on the same CPU")
    y = report.figure(y, "wall_time.png", 167)
    y = p(y + 4, "Figure 2. Lines join five-instance means. Dots are single runs. Both axes "
          "are logarithmic. Timers include solver work and exclude input generation and "
          "separate reference verification. The classical solver caps its swap shortlist on "
          "the extension. These timings describe a host implementation and a fixed shot budget.", 9, 13, MUTED)
    y = p(y + 12, f"At 1,000 assets the sampler averaged {float(summary[-2]['trained_seconds_mean']):.3f} s "
          f"against {float(summary[-2]['classical_seconds_mean']):.3f} s. At 10,000 the means were "
          f"{float(ext['trained_seconds_mean']):.2f} s and {float(ext['classical_seconds_mean']):.3f} s.")
    diff = [float(r["trained_minus_classical_bp"]) for r in off]
    wins, ties = sum(v < -1e-6 for v in diff), sum(abs(v) <= 1e-6 for v in diff)
    classical_hits = sum(float(r["classical_gap_bp"]) < 1e-6 for r in off)
    p(y + 8, f"The sampler matched {hits} official optima and the classical solver matched "
      f"{classical_hits}. Head to head, the sampler won {wins} cases, tied {ties} and lost "
      f"{len(off)-wins-ties}.", 10, 14)

    report.start("Controls")
    report.title("What did the optimizer learn?")
    y = p(114, "Three conditions used the same instance, starting state, random stream and "
          "shot budget. The frozen arm repeatedly measured its initial state. The schedule "
          "arm increased concentration and updated average factor exposure. The full arm "
          "also increased the weight of assets that appeared in its lowest-energy eighth.")
    y = h(y + 28, "Paired changes in best feasible score")
    rows = []
    for d in summary:
        rows.append([sized(d["n"]),
                     f"{float(d['frozen_minus_trained_bp_mean']):.2f} ± {float(d['frozen_minus_trained_bp_sd']):.2f}",
                     f"{float(d['schedule_minus_trained_bp_mean']):+.3f} ± {float(d['schedule_minus_trained_bp_sd']):.3f}",
                     f"{float(d['final_valid_pct_mean']):.1f}%"])
    y = t(y+16, ["Assets", "Frozen minus full (bp)", "Schedule minus full (bp)", "Final feasible shots"],
          rows, [75, 150, 160, BODY-385], height=26)
    y = p(y+8, "Positive differences favor the full arm. Each paired difference is computed "
          "within an instance before taking its mean and sample SD. The last column describes "
          "all 256 shots in the final full-arm batch, averaged over seeds.", 9, 13, MUTED)
    y = h(y+31, "Most of the gain came from changing the state")
    y = p(y, f"At 1,000 assets, the full arm gained {float(summary[-2]['frozen_minus_trained_bp_mean']):.2f} "
          "objective bp over the frozen distribution. Against the schedule arm its mean "
          f"score was {abs(float(summary[-2]['schedule_minus_trained_bp_mean'])):.3f} bp worse. "
          "The per-asset feedback gives no consistent gain across sizes. Strong initial "
          "preferences, factor updates and increased concentration explain much of the observed quality.")
    y = h(y+31, "Variation from the shot stream")
    for n in (200, 1000):
        per_instance = []
        effects = []
        for seed in range(5):
            reps = [r for r in repeats if r["n"] == n and r["seed"] == seed]
            if len(reps) != 5:
                raise ValueError("Missing replicate")
            per_instance.append(stdev(r["trained"]["objective"] * 1e4 for r in reps))
            effects.extend((r["schedule_only"]["objective"]-r["trained"]["objective"])*1e4 for r in reps)
        y = p(y+4, f"At {n:,} assets, five independent shot streams per fixed universe gave "
              f"a mean within-instance SD of {mean(per_instance):.3f} objective bp. The mean "
              f"paired schedule-minus-full difference across those 25 runs was {mean(effects):+.3f} bp.", 10.5, 15)
    p(y+9, "This secondary repeat study reuses the five generator seeds. The 25 trajectories "
      "measure optimizer variation on five markets.", 9, 13, MUTED)

    report.start("Circuit audit")
    report.title("Interference in the small circuits")
    y = p(112, "An address register names a portfolio. A cost phase uses its energy, then a "
          "Grover mixer changes its amplitude by interference with the uniform state. Two "
          "such layers share a fixed angle-search budget across encodings. The hard dictionary "
          "contains feasible addresses only. The penalty dictionary contains all K-asset baskets.")
    y = report.figure(y+16, "qaoa_optimum.png", 177)
    y = p(y+7, "Figure 3. Probability that a single ideal shot finds the true optimum. "
          "Thin lines connect results for one generator seed. Heavy lines show means. "
          "Uniform feasible sampling isolates the contribution of training within the hard dictionary.",
          9, 13, MUTED)
    qr = []
    for n in (10, 20):
        group = [r for r in qrows if r["n"] == n]
        for name, key in (("Feasible", "hard"), ("Penalty", "penalty")):
            arm = [r["cases"][key] for r in group]
            qr.append([str(n), name, f"{mean(v['uniform']['optimal_mass'] for v in arm)*100:.2f}%",
                       f"{mean(v['trained']['optimal_mass'] for v in arm)*100:.2f}%",
                       f"{mean(v['trained']['feasible_mass'] for v in arm)*100:.2f}%",
                       f"{mean(v['trained']['optimum_hit_16_shots'] for v in arm)*100:.2f}%"])
    y = t(y+25, ["Assets", "Dictionary", "Uniform hit", "Trained hit", "Trained valid", "Hit in 16 shots"],
          qr, [48, 84, 88, 87, 95, BODY-402], size=9.5)
    y = h(y+30, "A check on the quantum mechanism")
    err = max(c["max_probability_error"] for r in qrows for c in r["cases"].values())
    y = p(y, f"The hard encoding improved single-shot optimum probability on all ten tested "
          "universes. At 20 assets, 59–92 of 190 possible pairs were feasible. The Qiskit "
          "gate-state probabilities and the analytical Grover evolution had a maximum difference "
          f"of approximately {err:.3g} across 20 circuits. Separately, the 10-asset weighted-state "
          f"circuit's maximum difference from enumeration was about {max(r['max_probability_error'] for r in weighted):.3g} "
          "on the five trained parameter vectors.")
    y = p(y+14, "These compact circuits use 4–7 address qubits for the feasible dictionary and "
          "6–8 for the penalty dictionary. Preparing the dictionary scores every possible pair "
          "and already exposes the optimum. The measured probability gain isolates interference "
          "and constraint handling on known energies. It provides no evidence of faster discovery.", 10.5, 15, MUTED)

    report.start("Interpretation")
    report.title("Certificates and limits of inference")
    y = p(111, "A good benchmark needs a way to detect when two solvers both miss a better "
          "basket. The factor risk is a squared length. A tangent line beneath that square "
          "gives an underestimate for every portfolio at once. Combining it with nonnegative "
          "prices for the budget and sector caps gives a global lower bound.")
    y = p(y+10, "The returned feasible score is an upper bound on the minimum. When the "
          "distance between those two bounds is small, no unknown portfolio can improve much "
          "on the measured result. Reduced costs fix assets whose inclusion or exclusion would "
          "already exceed that full distance. The verifier can then enumerate the remaining "
          "candidates. METHODS.md derives the certificate and documents floating-point guards.")
    y = p(y+9, "The primary verifier resolved 32 of 40 optima. In a subsequent exhaustive "
          "check, each changed asset had to fit within the total space between the incumbent "
          "and the lower bound. Matching affordable additions and removals resolved the other "
          "eight, including four 10,000-asset cases. Exact verification is timed separately.")
    y = h(y+29, "Readout sensitivity under a hypothetical error model")
    y = p(y, "Suppose every output bit independently flips with probability 0.1%. The basket "
          "is unchanged only if all bits survive. Its size remains K if the numbers of false "
          "additions and deletions match. The latter can still change spending and sector use.")
    noise = []
    for n, k in ((10, 2), (1000, 100), (10000, 1000)):
        e = .001
        same = (1-e)**n
        # Adjacent probability terms have a stable ratio; this avoids large binomials.
        term, count_ok = same, same
        for j in range(min(k, n-k)):
            term *= ((k-j)*(n-k-j)/(j+1)**2) * (e/(1-e))**2
            count_ok += term
        noise.append([f"{n:,}", str(k), f"{100*same:.4f}%", f"{100*count_ok:.4f}%"])
    y = t(y+24, ["Assets", "Holdings", "Same bit string", "Still exactly K"], noise, [80, 85, 160, BODY-325])
    y = p(y+8, "Analytical illustration only. These percentages depend on the assumed independent "
          "flip rate and omit gate errors, correlations and device topology. All reported solver runs used "
          "ideal simulation as required by the participant guide.", 9, 13, MUTED)
    y = h(y+27, "Reproduction and sources")
    y = p(y, "The repository contains the pre-run plan, unchanged generator, returned holdings, "
          "Qiskit circuits, gate counts, seeds, per-round traces and a test suite. README.md "
          "lists the commands. The generator models hypothetical markets. Five universes per "
          "size support descriptive comparisons within this benchmark.")
    sources = [("Fixed-size weighted sampling", "https://doi.org/10.1093/biomet/81.3.457"),
               ("Lower-tail variational optimization", "https://quantum-journal.org/papers/q-2020-04-20-256/"),
               ("Feasible Grover mixers", "https://arxiv.org/abs/2006.00354"),
               ("Conditional-gradient certificates", "https://proceedings.mlr.press/v28/jaggi13.html")]
    p(y+10, "Primary sources include " + " &nbsp; / &nbsp; ".join(
        f'<link href="{url}" color="{ACCENT}">{escape(name)}</link>' for name, url in sources)
      + ". The research plan and methods appendix give the full references.", 9.5, 14, MUTED)
    report.save()


if __name__ == "__main__":
    main()
