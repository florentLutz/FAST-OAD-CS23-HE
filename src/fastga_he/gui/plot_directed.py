from bokeh.plotting import figure, show

if __name__ == "__main__":

    p = figure(width=300, height=300)
    p.multi_line(xs=[[1, 2, 3], [2, 3, 4]], ys=[[6, 7, 2], [4, 5, 7]],
                color=['red','green'])

    show(p)

