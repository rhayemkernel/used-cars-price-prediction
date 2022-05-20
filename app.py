from os import name
from flask import Flask,render_template,request,jsonify,url_for
from werkzeug.wrappers import response
import server.util

app = Flask(__name__,template_folder='client',static_folder='client')

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
   if request.method == 'POST':
      name = request.form['name']
      transmission = request.form['transmission']
      fuel = request.form['fuel']
      owner = int(request.form['owner'])
      year = int(request.form['year'])
      km_driven = int(request.form['km_driven'])
      engine = int(request.form['engine'])
      max_power = int(request.form['max_power'])
      response= jsonify({'estimated_price':server.util.predict_price(name,transmission,fuel,owner,year,km_driven,engine,max_power)})
      response.headers.add('Access-Control-Allow-Origin', '*')
      return response
@app.route('/get_car_name', methods=['GET'])
def get_car_name():
    response = jsonify({
        'car_name': server.util.get_car_name()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response  
if __name__ == '__main__':
   server.util.load_artifacts()
   app.run(debug=True)
