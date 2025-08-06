fig = px.sunburst(df1, path = ['Country', 'Genre'], values = 'Count', color = 'Country',
                 color_discrete_map = {'united states': '#85e0e0', 'india': '#99bbff', 'united kingdom': '#bfff80'})
fig = px.sunburst( df, path=['colA', 'colB'], value='colC')
fig.update_layout(title_text='Distribution of Genres in India, US, UK', title_x=0.5)                  
fig.show()
