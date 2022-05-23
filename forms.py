from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, URL

#WTForms
class UploadForm(FlaskForm):
    img_url = StringField("Blog Image URL", validators=[DataRequired(), URL()])
    submit = SubmitField("Upload Image")