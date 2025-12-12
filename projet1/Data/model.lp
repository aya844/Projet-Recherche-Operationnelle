\ Model Organs_Transport_MILP
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xec82e2d58374a1e2
Minimize
  7.3 x_H1_C1_kidney + 11.1 x_C1_H2_kidney + 5 n_H1_C1 + 5 n_C1_H2
Subject To
 flow_H1_kidney: x_H1_C1_kidney = 3
 flow_C1_kidney: - x_H1_C1_kidney + x_C1_H2_kidney = 0
 flow_H2_kidney: - x_C1_H2_kidney = -2
 cap_H1_C1: x_H1_C1_kidney - 2 n_H1_C1 <= 2
 cap_C1_H2: x_C1_H2_kidney - 2 n_C1_H2 <= 1
Bounds
Generals
 n_H1_C1 n_C1_H2
End
