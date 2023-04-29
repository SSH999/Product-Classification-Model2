import csv

data = [["This is a great phone for everyday use", "electronics"],
        ["This camera takes amazing photos", "electronics"],
        ["These shoes are very comfortable and stylish", "clothing"],
        ["I love this book, it's a great read", "books"],
        ["This is a very effective cleaning product", "home and kitchen"],
        ["I bought this toy for my child and they love it", "toys"],
        ["These headphones have great sound quality", "electronics"],
        ["I use this software for work and it's very helpful", "computers"],
        ["This cookware set is durable and easy to clean", "home and kitchen"],
        ["This shirt fits well and is made of high-quality material", "clothing"]]

with open('product_data1.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['product_description', 'category'])
    for row in data:
        writer.writerow(row)
