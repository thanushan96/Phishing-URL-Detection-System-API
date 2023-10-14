from flask_cors import CORS
from flask import Flask
import database
from routes.auth import auth_bp
from routes.predict import predict_bp

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {"origins": "http://localhost:3000"},
    r"/login": {"origins": "http://localhost:3000"},
    r"/register": {"origins": "http://localhost:3000"}
})


# Initialize the database 
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost/DB'
app.secret_key = 'c3c276452481cef3fae39c2de9475431f1c5ea781dad07be097be89fe0666e09'
 
database.initialize_database(app)


#%%
# Register the blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(predict_bp)

#%%


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=False)
