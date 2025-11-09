# # Python program for DDA line generation By Ranjit Das

from matplotlib import pyplot as plt


def plot_line(x,y):
    plt.plot(x, y, marker="o", markersize=3, markerfacecolor="black")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("DDA Line Algorithm")
    plt.savefig("aa.png")


def DDA(x0, y0, x1, y1):
    
    del_x = abs(x1) - abs(x0)
    del_y = abs(y1) - abs(y0)
    steps = max(del_x,del_y)
    print("No of steps:",steps)
    
    x_inc = abs(del_x/steps)
    # print("x value:",x_inc)
    
    y_inc = abs(del_y/steps)
    # print("y value:",y_inc)
    
    # make a list for coordinates
    x_coorinates = []
    y_coorinates = []
    # print("Loop starts here:")
    for i in range(steps):
        x0 += x_inc
        y0 += y_inc
        x_coorinates.append(int(x0))
        y_coorinates.append(int(y0))
        
    print()
    plot_line(x_coorinates,y_coorinates)
    for i,j in zip(x_coorinates, y_coorinates):
        print(f"{i},{j}")


# Driver code
if __name__ == "__main__":

    # coordinates of 1st point
    x0, y0 = 10, 50

    # coordinates of 2nd point
    x1, y1 = 50, 100


    # Function call
    DDA(x0, y0, x1, y1)
    # This code is contributed by 111arpit1


