from flask import Flask, render_template, request
from friedpickles import get_quote

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        quotes = get_quote(prompt)
        return render_template('result.html', quotes=quotes)
    return render_template('home.html')

if __name__ == '__main__':
    app.run()