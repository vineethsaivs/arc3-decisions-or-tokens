"""What verify's helpers reported in the three ustv runs (run from repo root): check_model reports that matched every
checked transition vs found a mismatch, time-limit stops, check_goal reports and how many found a counterexample."""
import glob, re
tot = [0] * 8
for i in (1, 2, 3):
    t = "".join(open(f, errors="replace").read()
                for f in glob.glob(f"solution/kaggle/generated/kernel-outputs/kaggle-lf15ustv-1101-{i}/*_p0.txt"))
    cm = re.findall(r"check_model: (\d+) of (\d+) checked transitions", t)
    # each non-matching report: does its shown mismatch list name a wrong prediction, or only exceptions from model code?
    reps = re.findall(r"check_model: (\d+) of (\d+) checked transitions.*?(?:First mismatches \(newest first\): ([^\n]*))?(?=\n|$)", t)
    wrong = sum(1 for a, b, m in reps if a != b and "predicted" in m)
    only_err = sum(1 for a, b, m in reps if a != b and m and "predicted" not in m)
    full = sum(1 for a, b in cm if a == b and b != "0")
    goal_false = len(re.findall(r"check_goal: false on all \d+", t))
    goal_hit = len(re.findall(r"check_goal: true on \d+ of", t))
    row = [len(cm), full, len(cm) - full, t.count("stopped at the time limit"), goal_false, goal_hit, wrong, only_err]
    tot = [a + b for a, b in zip(tot, row)]
    print(f"run {i}: check_model reports {row[0]}, all match {row[1]}, mismatch found {row[2]}, time-limit stops {row[3]}; "
          f"check_goal false-on-all {row[4]}, counterexample {row[5]}; of the mismatches: wrong prediction {row[6]}, only errors {row[7]}")
print(f"total: check_model {tot[0]} (all match {tot[1]}, mismatch {tot[2]}, time-limit {tot[3]}); check_goal {tot[4] + tot[5]} "
      f"(counterexample {tot[5]}); mismatch reports naming a wrong prediction {tot[6]}, only errors from model code {tot[7]}")
