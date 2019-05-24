import os
import sys
import importlib

from flask import render_template, flash, request, jsonify, url_for, current_app
from flask_login import login_required


from app.reports import bp
from app.reports.forms import ProcessTaskForm
from app.models import Acquisition, Fact, ProcessTask


@bp.route('/report/<acquisition_uuid>',  methods=['GET', 'POST'])
@login_required
def report(acquisition_uuid, pending_id=None):

    acquisition = Acquisition.query.filter_by(id=acquisition_uuid).first_or_404()
    from app.tasks import produce_report
    if request.method == 'GET':

        form = ProcessTaskForm()

        tasks = [(x.name, x.name) for x in ProcessTask.query.all()]

        form.process_task_name.choices = tasks

        facts = Fact.query.filter_by(acquisition_id=str(acquisition.id)).all()

        if pending_id:
            pending = produce_report.AsyncResult(pending_id)
        else:
            pending = None

        if not facts:
            flash(f'No reports found for {acquisition.id.hex}', 'danger')

        return render_template('report.html',
                               form=form,
                               acquisition=acquisition,
                               facts=facts,
                               tasks=tasks,
                               pending=pending)

    elif request.method == 'POST':
        current_app.logger.info('hello')
        fn = request.form['process_task_name']
        current_app.logger.info(fn)
        res = produce_report.delay(fn, {'id': acquisition.id,
                                        'hex': acquisition.id.hex,
                                        'author_id': acquisition.author.id,
                                        'author_hex': acquisition.author.id.hex,
                                        'files': acquisition.files
                                        }
                                   )
        current_app.logger.info(res)
        flash(f'Starting process: {fn}', 'info')

        return url_for('reports.report', acquisition_uuid=acquisition.id, pending=res.id)






