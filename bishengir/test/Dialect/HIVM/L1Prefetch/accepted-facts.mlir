// RUN: python3 %S/Inputs/check_accepted_facts.py bishengir-opt %S
// All schedules publish only after proof, survive serialization/reanalysis,
// reject tampering before transformation, and retire the entire private record.
