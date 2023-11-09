class EmptyRowException(Exception):

    msg = '找不到水平線'

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class EmptyColException(Exception):

    msg = '找不到垂直線'

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

