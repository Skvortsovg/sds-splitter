class BaseSplitterException(Exception):
    pass


class ErrorTrainTestSplit(BaseSplitterException):
    def __init__(self, stage, message):
        self.stage = stage
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.stage} {self.message}"


class StratifiedClassHasOneMember(ErrorTrainTestSplit):
    def __init__(self, stage, message=None):
        message = message or "Some class from stratification_column have only 1 member"
        super().__init__(stage, message)
