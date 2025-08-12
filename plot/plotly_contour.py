# 5. 2D Density Contour Plot with Low Density Contours
fig5 = px.density_contour(df, x='x', y='y',
                          title='2D Density Contour Plot with Low Density Contours')

# Update contours to show more levels including low density areas
fig5.update_traces(
    contours=dict(
        start=0,           # Start from lowest density
        end=None,          # Go to maximum density (auto)
        size=None,         # Auto-determine step size
        showlines=True,    # Show contour lines
        showlabels=True,   # Show density values on contours
        labelfont=dict(size=10, color="black")
    ),
    ncontours=15,         # Increase number of contour levels (default is ~10)
    line=dict(width=1)    # Make contour lines thinner for better visibility
)
