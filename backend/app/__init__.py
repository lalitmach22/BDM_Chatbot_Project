from flask import Flask
from app.config import Config
from app.extensions import db, cors

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize extensions
    db.init_app(app)
    cors.init_app(app)

    # Register blueprints
    from app.routes import blueprint
    app.register_blueprint(blueprint)

    return app

