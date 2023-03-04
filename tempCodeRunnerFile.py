   def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        print(self.size())
        return super().resizeEvent(event)