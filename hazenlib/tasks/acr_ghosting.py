from hazenlib.HazenTask import HazenTask


class ACRGhosting(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        results = {'Not implemented yet': {'Error': self}}
        return results
