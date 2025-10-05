import sys

if len(sys.argv) != 4:
    print("Uso: mi_script.py <param1> <param2> <param3>")
    sys.exit(1)

param1 = sys.argv[1]
param2 = sys.argv[2]
param3 = sys.argv[3]

print(f"Python recibio: {param1}, {param2}, {param3}")