import os
import shutil

from flask import request, render_template, url_for, redirect, flash, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import pydicom.errors

from app.uploader import bp
from app.models import Acquisition, User


class SeriesExistsError(Exception): pass


@bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():

    if request.method == 'POST':

        for key, f in request.files.items():

            if key.startswith('file'):

                secure_path = os.path.join(current_app.config['UPLOADED_PATH'], secure_filename(f.filename))
                f.save(secure_path)

                try:
                    filesystem_dir = ingest(secure_path)
                except SeriesExistsError:
                    os.remove(secure_path)
                    flash('SeriesInstanceUID already exists!')
                    return redirect(url_for('main.user', username=current_user.username))

                permanent_path = os.path.join(filesystem_dir, secure_filename(f.filename))

                shutil.move(secure_path, permanent_path)
                flash('Upload success!')

        return redirect(url_for('main.user', username=current_user.username))

    return render_template('upload.html')


def ingest(f):
    try:
        dcm: pydicom.Dataset = pydicom.read_file(f)
        series_instance_uid = dcm.SeriesInstanceUID
        description = f"{dcm.StudyDescription}: {dcm.SeriesDescription}"
        files = 1

        if Acquisition.query.filter_by(series_instance_uid=series_instance_uid).first():
            current_app.logger.info('series exists')
            raise SeriesExistsError(f"UID: {series_instance_uid}")

        acq = Acquisition(series_instance_uid=series_instance_uid,
                          files=files,
                          description=description,
                          user_id=current_user.get_id())
        acq.save()

        user = User.query.get(current_user.get_id())
        directory = os.path.join(current_app.config['UPLOADED_PATH'], user.filesystem_key, acq.filesystem_key)
        os.makedirs(directory, exist_ok=True)

        return directory

    except Exception as e:
        raise

