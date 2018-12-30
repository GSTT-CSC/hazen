import os

from flask import request, render_template, url_for, redirect, flash, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from app.uploader import bp


@bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        f.save(os.path.join(current_app.config['UPLOADED_PATH'], secure_filename(f.filename)))
        flash('Upload success!')
        return redirect(url_for('main.user', username=current_user.username))
    return render_template('upload.html')
