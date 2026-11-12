import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from g4d_readings import read_d1, read_d2, read_d3, use_rule
assert read_d1(75).startswith("75 or below") and read_d1(76).startswith("Between") and read_d1(81).startswith("Between") and read_d1(82).startswith("82 or above")
assert read_d3(75).startswith("At least 6") and read_d3(74).startswith("Within 5") and read_d3(64).startswith("Within 5") and read_d3(63).startswith("More than 5")
assert read_d2(82).startswith("82 or above") and read_d2(81).startswith("76 to 81") and read_d2(76).startswith("76 to 81") and read_d2(75).startswith("75 or below")
assert use_rule(82, 75).startswith("COME BACK") and use_rule(81, 75).startswith("Format stays") and use_rule(82, 76).startswith("Format stays") and use_rule(60, 60).startswith("Format stays")
print("test_g4d_readings: OK")
