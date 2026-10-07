"""Stock Market Prediction - Flask application factory."""
from __future__ import annotations

import logging

from flask import Flask, render_template


def create_app() -> Flask:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)

    from .routes import bp
    app.register_blueprint(bp)

    @app.template_filter("money")
    def money(value) -> str:
        return "N/A" if value is None else f"₹{value:,.2f}"

    @app.template_filter("volume")
    def volume(value) -> str:
        return "N/A" if value is None else f"{value:,}"

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", title="Page not found",
                               message="We couldn't find that page or stock."), 404

    return app
