version 1.0
# cQASM generated by QI backend for Qiskit
qubits 4
H q[0]
H q[1]
H q[2]
H q[3]
CNOT q[0], q[1]
Rz q[1], -1.151900
CNOT q[0], q[1]
CNOT q[0], q[2]
CNOT q[1], q[3]
Rz q[2], -7.055385
CNOT q[0], q[2]
Rz q[3], -3.743674
CNOT q[1], q[3]
CNOT q[3], q[2]
Rz q[2], -10.619074
CNOT q[3], q[2]
Rx q[0], 10.244043
Rx q[1], 10.244043
Rx q[2], 10.244043
Rx q[3], 10.244043
measure q[0]
measure q[1]
measure q[2]
measure q[3]
