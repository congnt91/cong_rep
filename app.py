from flask import Flask, render_template

try:
    from vnstock import Vnstock
    vn = Vnstock()
except Exception as e:
    vn = None
    vn_import_error = e

app = Flask(__name__)

@app.route('/')
def index():
    """Homepage that fetches stock data using vnstock library."""
    error = None
    data = None
    if vn is None:
        error = f"Could not import vnstock: {vn_import_error}"
    else:
        try:
            # Fetch sample data for ticker ACB using source VCI
            df = vn.stock(symbol="ACB", source="VCI").price_board()
            data = df.to_dict(orient="records")
        except Exception as e:
            error = str(e)
    return render_template('index.html', data=data, error=error)

if __name__ == '__main__':
    app.run(debug=True)
