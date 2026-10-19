"""Needleman–Wunsch global alignment: returns the optimal score only."""
MATCH=2
MISMATCH=-1
GAP=-2

def needleman_wunsch(seq1, seq2):
    """Calculate a global alignment score with a full dynamic-programming matrix."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i * GAP
    for j in range(1, n + 1):
        dp[0][j] = j * GAP

    for i in range(1, m + 1):
        previous, current = dp[i - 1], dp[i]
        for j in range(1, n + 1):
            diagonal = previous[j - 1] + (
                MATCH if seq1[i - 1] == seq2[j - 1] else MISMATCH
            )
            up = previous[j] + GAP
            left = current[j - 1] + GAP
            current[j] = max(diagonal, up, left)

    return dp[m][n]
