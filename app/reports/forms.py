from flask_wtf import FlaskForm
from wtforms import PasswordField, BooleanField, SubmitField, StringField, SelectField
from wtforms.validators import ValidationError, DataRequired


class ProcessTaskForm(FlaskForm):

    process_task_name = SelectField('Process Task',
                                    render_kw={"onchange": "this.form.submit();"})


