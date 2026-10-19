"""Smith–Waterman local alignment: returns the optimal score only."""
MATCH=2
MISMATCH=-1
GAP=-2

def smith_waterman(seq1, seq2):
    """Calculate a local alignment score with a full dynamic-programming matrix."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best_score = 0

    for i in range(1, m + 1):
        previous, current = dp[i - 1], dp[i]
        for j in range(1, n + 1):
            diagonal = previous[j - 1] + (
                MATCH if seq1[i - 1] == seq2[j - 1] else MISMATCH
            )
            up = previous[j] + GAP
            left = current[j - 1] + GAP
            current[j] = max(0, diagonal, up, left)
            if current[j] > best_score:
                best_score = current[j]

    return best_score
