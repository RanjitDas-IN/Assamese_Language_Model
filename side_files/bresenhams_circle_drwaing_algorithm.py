from matplotlib import pyplot as plt
import math

def plot_circle(x, y, filename="aa.png"):
    plt.figure(figsize=(5,5))
    plt.scatter(x, y,s=5,edgecolors="black",c="red")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Bresenham's Circle Drawing Algorithm")
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")



def get_circle_ohter_points_after_algorithm(xc, yc, x_list, y_list):
    x_points = []
    y_points = []
    for x, y in zip(x_list, y_list):
        pts = [
            (xc + x, yc + y),
            (xc - x, yc + y),
            (xc + x, yc - y),
            (xc - x, yc - y),
            (xc + y, yc + x),
            (xc - y, yc + x),
            (xc + y, yc - x),
            (xc - y, yc - x),
        ]
        for px, py in pts:
            x_points.append(px)
            y_points.append(py)
    return x_points, y_points






def bresenhams_circle(r):
    x_coords = []
    y_coords = []
    x = 0
    y = r
    p = 3 - 2 * r
    x_coords.append(x)
    y_coords.append(y)
    while x < y:
        x += 1
        if p <= 0:
            p = p + 4 * x + 6
        else:
            y -= 1
            p = p + 4 * (x - y) + 10
        x_coords.append(x)
        y_coords.append(y)
    return x_coords, y_coords


if __name__ == "__main__":
    center_x, center_y = 0, 0
    radius = 25
    x_c, y_c = bresenhams_circle(radius)
    x_points, y_points = get_circle_ohter_points_after_algorithm(center_x, center_y, x_c, y_c)

    
    seen = set()
    uniq_x = []
    uniq_y = []
    for xp, yp in zip(x_points, y_points):
        if (xp, yp) not in seen:
            seen.add((xp, yp))
            uniq_x.append(xp)
            uniq_y.append(yp)

    print("Number of plotted points:", len(uniq_x))
    # print("x_points =", uniq_x)
    # print("y_points =", uniq_y)
    plot_circle(uniq_x, uniq_y)
    
