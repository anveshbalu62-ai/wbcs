import sys
with open("final_out.txt", "w", encoding="utf-8") as f:
    sys.stdout = f
    sys.stderr = f
    import test_predict
