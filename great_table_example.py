quarterly_comparison = pd.DataFrame({
    'Region': ['North', 'South', 'East', 'West'],
    'Q1_Revenue': [450_000, 380_000, 520_000, 410_000],
    'Q1_Profit': [67_500, 53_200, 78_000, 61_500],
    'Q2_Revenue': [485_000, 395_000, 548_000, 430_000],
    'Q2_Profit': [72_750, 55_300, 82_200, 64_500],
})

comparison_table = (
    GT(quarterly_comparison)
    .tab_header(title="Regional Financial Performance")
    .tab_spanner(label="Q1 2025", columns=['Q1_Revenue', 'Q1_Profit'])
    .tab_spanner(label="Q2 2025", columns=['Q2_Revenue', 'Q2_Profit'])
    .cols_label(
        Q1_Revenue='Revenue',
        Q1_Profit='Profit',
        Q2_Revenue='Revenue',
        Q2_Profit='Profit'
    )
    .fmt_currency(
        columns=['Q1_Revenue', 'Q1_Profit', 'Q2_Revenue', 'Q2_Profit'],
        currency='USD',
        decimals=0
    )
)
