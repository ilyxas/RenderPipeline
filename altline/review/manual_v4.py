"""Reviewer-verified findings for the v4 audit (written by hand after reading the data and the
preview frames; merged into reports/v4_<clip>/dev_audit.json by audit_v4.run via manual.json).

    python3 -m altline.review.manual_v4   # writes reports/v4_<clip>/manual.json
"""

from altline.io_utils import REPORTS_ROOT, write_json

LEVEL = ("implemented + functionally verified (all stages run, bundle under the gate by construction); "
         "quality NOT accepted")

M = {
    "061c": {
        "verdict": "partial: stability better than v3 at the damper input (6 joints 105 -> 10 frames), no visible "
                   "regression; remaining peaks are internal; chest yaw 33 deg (C3) unchanged",
        "verified_findings": [
            "elbow_r frames 83-92: twist split. Elbow bend crosses the 5-20 deg twist window (debiased 18 -> 2.5 -> 15 deg); "
            "upper-arm twist 19 -> 0 -> 28 deg, forearm roll counter-rotates; per-frame local acc 14-16k while the solved "
            "forearm world frame is 3-6x lower. Internal (solver).",
            "elbow_r/wrist_r frames 96-103: calibration flattens the right forearm depth: filtered dz +0.08 m -> calibrated "
            "-0.002 m from frame 94, forearm stretched to 1.15 L (C4). Raw tracker palm turns 63 deg smoothly over frames "
            "94-104; after calibration it turns ~100 deg within frames 98-102. Damper input 14,059 deg/s^2 at f101, damper "
            "deviation 15.9 deg. Internal (calibration).",
            "Preview frames 89, 98-105, 120 checked: stick figure matches v3 and the tracker dots; forearm roll is not "
            "visible in the stick preview (needs a skinned mesh)."],
    },
    "0663": {
        "verdict": "fail (pose): hands still 27/23 cm from the opposite shoulder, forearms intersect (C6); "
                   "stability: C1 fixed, remaining peaks mostly real tracker noise when the arms cross",
        "verified_findings": [
            "C1 fixed: v3 solver input had depth-sign flips at frames 28, 31 (forearm_l), 32 (upper_arm_l), 39 (upper_arm_r) "
            "with jumps 17-23 cm; v4 has no flips, max jump 8.3 cm. Preview frame 28: v3 left arm swings behind the body, "
            "v4 follows the tracker dots in front.",
            "elbow_l f28 / wrist_l f29 / elbow_l f33 / elbow_r f38: tracking data. Raw tracker forearm direction acc 35k at "
            "f28 and the palm reading jumps 59 deg in one frame while the arms cross; filter + depth reduce it to 9.5k at the "
            "solver input, 4.8k at the damper input. Damper deviations 12-16 deg, hand tips moved up to 5.8 cm.",
            "elbow_l f22 (twist split, x2.1) and elbow_r f25 / wrist_l f34 (calibration, x3.3-3.4): internal.",
            "Arm peak speed (image chord) 391 -> 248 deg/s on the right arm (-37 % vs source 2D); the loss happens before "
            "the solver (385 per-frame raw -> 275 after filter+calibration). Excursion is kept (139 vs 143 deg).",
            "C6 not fixed: forearms min distance 1.2 cm, 104 frames under 5 cm."],
    },
    "5313": {
        "verdict": "good for stability: nothing over the gate at the damper input, damper did nothing; head -50.9 vs "
                   "-49.4 deg (2D) as in v3",
        "verified_findings": [
            "C2 symptom gone here: v3 upper_arm_l peak 11,205 at f25; v4 per-frame 7,612 (filtered input), damper input 1,575.",
            "Frames over the gate (6 joints): BEFORE 33 -> per-frame 7 -> temporal 0. Damper: 0 frames, 0 deg."],
    },
    "619b": {
        "verdict": "stability good (0 frames at the damper input, damper ~0 deg); head yaw still wrong (C5): rig +21.5 deg "
                   "vs image -24 deg",
        "verified_findings": [
            "Frames over the gate (6 joints): BEFORE 88 -> per-frame 18 -> temporal 0; damper touched 6 bone-frames, "
            "deviation < 0.1 deg.",
            "C5 not fixed: head yaw hold median +21.5 deg (v3 +20.5) against -24.0 from the image; error 45.6 deg."],
    },
    "690b": {
        "verdict": "good overall; stability 134 -> 4 frames at the damper input (damper max 5 deg); left forearm "
                   "slightly worse in 2D than v3 (p95 6.4 -> 10.1 deg) because of the 5-20 deg twist window",
        "verified_findings": [
            "elbow_l f18: calibration flattens the left forearm depth: filtered dz +0.100 m, calibrated 0.041 (f17) -> "
            "0.000 (f18) while the arm rises; forearm 1.19 L (C4). Upper-arm twist -54 -> -87 deg in one frame. "
            "Landmark forearm acc 384 (filtered) -> 6,318 (calibrated). Internal (calibration).",
            "Twist-window ablation: v4a (2-10 deg) forearm_l 2D p95 6.5 deg, v4 (5-20 deg) 10.1 deg."],
    },
    "85a9": {
        "verdict": "fail (pose): palms 23/40 cm from the face (worse than v3 16/36), chest yaw 32 deg (C3); "
                   "remaining jerks are the C2 twist split",
        "verified_findings": [
            "elbow_l f15-17 / elbow_r f16-18: C2 twist split still in the per-frame solver. Debiased elbow bend runs "
            "5 -> 17 deg in 2 frames (raw 20 -> 26: the hinge de-bias doubles the slope), so the bend-plane twist weight "
            "goes 0 -> 1 and the upper-arm twist jumps 0 -> -72 deg (left) / 9 -> 84 deg (right) in 2-3 frames while the "
            "forearm counter-rotates. Local acc 21k vs world 2.4k (x8.5). Raw tracker under the gate (2.1-2.3k). "
            "Damper input 6-7k, deviations 12.9-18.7 deg, hand tips moved up to 8.8 cm.",
            "Not visible in the stick preview (frames 15-17, 22 identical to v3); on a skinned mesh it is a 70 deg "
            "upper-arm roll in 0.1 s.",
            "C4 worse: hand_l tip to face 16.4 -> 22.8 cm, hand_r 35.6 -> 39.9 cm (v4raw 21.1 / 39.7: mostly from non-Phase-2 "
            "code since v3). Preview f100: palms in front of the face as in v3.",
            "Arm peak speed 245 -> 144 deg/s (right, -41 % vs source 2D), mostly before the solver (225 raw per-frame -> 145)."],
    },
    "9645": {
        "verdict": "good for stability: nothing over the gate at the damper input, damper did nothing; head 43.3 vs "
                   "44.7 deg (2D)",
        "verified_findings": [
            "C2 symptom gone here: v3 upper_arm_r peak 12,049 at f56; v4 per-frame 6,886, damper input 1,912.",
            "Frames over the gate (6 joints): BEFORE 29 -> per-frame 6 -> temporal 0. Damper: 0 frames."],
    },
    "df1e": {
        "verdict": "fail, and the LEFT ARM IS WORSE THAN v3: from frame ~92 the left forearm hangs ~15 cm below the "
                   "tracker's wrist; the 53 deg damper deviation hides an internal artifact chain, not tracking data. "
                   "Leg: C1 fixed, knee still bent 48 deg (C4)",
        "verified_findings": [
            "Raw tracker is smooth here: left elbow bend 27-39 deg, bend-plane step <= 14 deg/frame, forearm 0.09-0.15 m "
            "toward the camera (frames 84-101).",
            "Step 1 (calibration, C4): from frame 92 the forearm depth is flattened: filtered dz +0.10 m -> calibrated "
            "0.0015 m, length 1.13 L; elbow bend 32 -> 10 deg; bend plane turns 128 deg in one frame (guarded). Forearm "
            "landmark acc 5.0k (filtered) -> 14.5k (calibrated).",
            "Step 2 (per-frame solver): upper-arm twist -44 -> +90 -> +161 -> 179 deg in frames 92-95, forearm roll "
            "+47 -> -86 -> -157 -> -176 deg; local acc 87,022 vs solved forearm world 14,475 (x6): a 180 deg twist split, "
            "humerus turned half a revolution.",
            "Step 3 (limits): forearm roll clamped to -90 with unwrap, excess pushed into the upper arm (-138 -> -269 deg, "
            "frames 100-114): forearm direction moves 27-33 deg away from the solver, hand world orientation up to 175 deg "
            "off (f93), forearm 158 deg off.",
            "Step 4 (damper): smooths each bone's local rotation on its own, so the world forearm ends up to 173 deg off "
            "(f95); max damper deviation 52.9 deg (forearm_l f92), upper_arm_l 43.4 deg with 29 frames over 10 deg; "
            "hand tip 8.5 cm.",
            "Visible: preview frames 94, 104, 110: v4 left forearm bends down to ~1.0 m height while the tracker wrist "
            "dots are at ~1.15 m and the actor's arm is horizontal; v3 follows the dots. 2D forearm_l p95 10.2 (v3) -> "
            "23.8 deg (v4), frames 101-117 at 15-29 deg; wrist 2D p95 5.0 -> 10.7 cm.",
            "Cause of the regression: twist window 5-20 deg. Ablation v4a (2-10 deg, all Phase 2 on): forearm_l p95 10.2 "
            "deg, wrist 5.1 cm, i.e. v3 level; v4w/v4 (5-20): 23.8 / 10.7.",
            "Leg: C1 fixed (v3 shin_l jump 21.4 cm + sign flip f86 -> v4 8.1 cm, no flip); ankle_l f82: raw tracker acc "
            "16k removed by the filter (2.8k), re-raised to 8.2k by the per-frame solver (x2.2), 3.6k at the damper "
            "input, foot moved 0.9 cm. Kick peak speed 473 (source 2D) -> 365 deg/s (temporal solver 426 -> 359).",
            "Event elbow_l f87 is labelled 'tracking data' by the automatic rule, but its damper deviation (36.8 deg) "
            "belongs to the f92 chain."],
    },
}


def main():
    for clip, m in M.items():
        write_json(REPORTS_ROOT / f"v4_{clip}" / "manual.json", {"reviewer_level": LEVEL, "reviewer_verdict": m["verdict"],
                                                                "reviewer_verified_findings": m["verified_findings"]})
        print("ok", clip)


if __name__ == "__main__":
    main()
