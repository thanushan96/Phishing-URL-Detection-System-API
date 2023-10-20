from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()



class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

def initialize_database(app):
    app.app_context().push()
    db.init_app(app)
    with app.app_context():
        db.create_all()
       




