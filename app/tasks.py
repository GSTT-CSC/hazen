import os

from flask import current_app, flash, jsonify

from app.models import ProcessTask, Fact

from hazen import worker


@worker.task(bind=True)
def produce_report(self, fn, acquisition):
    current_app.logger.info('hello')
    task = __import__(f'hazenlib.{fn}', globals(), locals(), [f'{fn}'])
    current_app.logger.info(task)

    process = ProcessTask.query.filter_by(name=fn).first()

    filesystem_folder = os.path.join(current_app.config['UPLOADED_PATH'],
                                     acquisition['author_hex'],
                                     acquisition['hex'])

    dcms = [os.path.join(filesystem_folder, f) for f in os.listdir(filesystem_folder)]

    if len(dcms) != acquisition['files']:
        raise Exception('Number of dicoms in directory not equal to expected!')

    self.update_state(state='IN PROGRESS')
    self.acquistion_id = acquisition['hex']
    res = task.main(data=dcms)
    self.update_state(state='STORING RESULTS')
    fact = Fact(user_id=acquisition['author_id'],
                acquisition_id=acquisition['id'],
                process_task=process.id,
                process_task_variables={},
                data=res)

    fact.save()
    flash(f'Completed process: {fn}')
    return jsonify(fact.data)