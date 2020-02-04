import os
import pkgutil
import importlib

from flask import current_app

from app import db, create_app, create_celery_app
from app.models import User, Acquisition, ProcessTask


__version__ = '0.1.dev0'
__author__ = "mohammad_haris.shuaib@kcl.ac.uk"


def register_tasks_in_db():
    import hazenlib
    tasks = {f'{modname}': importlib.import_module(f'hazenlib.{modname}') for importer, modname, ispkg in pkgutil.iter_modules(hazenlib.__path__)}

    with app.app_context():
        stored_tasks = ProcessTask.query.all()

        for stored_task in stored_tasks:

            if stored_task.name in tasks.keys():
                _ = tasks.pop(stored_task.name)
                current_app.logger.info(f'{stored_task.name} already exists in db')

        for name, obj in tasks.items():
            docstring = obj.__doc__.replace('\n', '\\n') if obj.__doc__ else 'No description available.'
            process_task = ProcessTask(name=name,
                                       docstring=docstring)
            process_task.save()


app = create_app()
worker = create_celery_app(app)
register_tasks_in_db()


@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'Acquisition': Acquisition, 'ProcessTask': ProcessTask}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))



