from hazenlib.HazenTask import HazenTask


class Relaxometry(HazenTask):

    def __init__(self, data_paths: list, report=False, report_dir=os.getcwd()):
        super().__init__(data_paths, report=report)

    def run(self):
        results = {'Not implemented yet': self}
        return results
