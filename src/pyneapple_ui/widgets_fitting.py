from __future__ import annotations
from typing import Callable
from PyQt6 import QtWidgets, QtCore
import numpy as np
from abc import abstractmethod


class WidgetData:
    """
    Basic widget enhancement class. To set up different dlg Widgets in the same way.

    Attributes:
    ----------
    name: str
        Name of the Widget and text that is displayed on the dlg ':' is  added separately
    value: int | float | np.ndarray | str
        The value the widget currently hold
    value_range: list
        Range of allowed Values
    value_type: Class | None = None
        Defines the value type of the handled variable.
        If the type is not defined here the input type of current_value will be used.
    tooltip: str | None = None
        Widget tooltip text
    value: @Property
        Can hold different types of classes to report back to main UI

    """

    def __init__(
        self,
        value: int | float | np.ndarray | str = 1,
        range_: list | np.ndarray | None = None,
        dtype: type = None,
    ):
        self.current_value = value
        self.range = range_ if not None else list()
        self.dtype = dtype if not None else type(value)
        if dtype is None:
            self.dtype = type(value)
        elif dtype is not None:
            self.dtype = dtype
        self._value = value
        self.alignment_flag = None

    @property
    def value(self):
        value = self._value
        value = self.value_to_dtype(value)
        return value

    @value.setter
    def value(self, value):
        # value = self.value_to_dtype(value)
        # value = self.value_check_range(value)
        self.update_widget(value)
        self._value = value

    def value_to_dtype(self, value):
        if isinstance(value, str):
            if not value.lower() == "none":
                if self.dtype == int or self.dtype == float:
                    if value.isdigit():
                        value = self.dtype(value)
                    # else:
                    # value = None
                elif self.dtype == np.ndarray:
                    try:
                        # value = np.from buffer(value)
                        if "[" in value:
                            value = value.replace("[", "")
                        if "]" in value:
                            value = value.replace("]", "")
                        value = np.fromstring(value, sep=" ")
                    except TypeError:
                        Warning(TypeError())
                # elif not self.dtype == bool:
                #     value = None
                elif isinstance(self.dtype, (list, str)):
                    pass
        return value

    @staticmethod
    def value_to_string(value):
        if not isinstance(value, str):
            value = str(value)
        return value

    @abstractmethod
    def update_widget(self, value):
        pass

    def value_check_range(self, value):
        if isinstance(value, (int, float)):
            if self.range[0] <= value <= self.range[1]:
                return value
            elif self.range[0] < value:
                print("Warning: Value exceeded range Limits.")
                return self.range[0]
            elif self.range[1] > value:
                print("Warning: Value exceeded range Limits.")
                return self.range[1]
        else:
            return value


class EditField(WidgetData, QtWidgets.QLineEdit):
    """QLineEdit enhanced with WidgetData"""

    def __init__(
        self,
        value: int | float | np.ndarray,
        range_: list | None,
        dtype: type | None = None,
        tooltip: str | None = None,
    ):
        WidgetData.__init__(self, value, range_, dtype)
        QtWidgets.QLineEdit.__init__(self)
        self.setText(str(value))
        self.textChanged.connect(self._text_changed)
        self.setMaximumHeight(28)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.value = value
        if tooltip:
            self.setToolTip(tooltip)

    def _text_changed(self):
        self.value = self.text()

    # def update_widget(self, value):
    #     pass

    def update_widget(self, value):
        value = self.value_to_string(value)
        if not value == self.text():
            self.setText(value)


class CheckBox(WidgetData, QtWidgets.QCheckBox):
    """QCheckbox enhanced with WidgetData"""

    def __init__(
        self,
        value: bool,
        range_: list,
        dtype: type | None = None,
        tooltip: str | None = None,
    ):
        WidgetData.__init__(self, value, range_, dtype)
        QtWidgets.QCheckBox.__init__(self)
        # self.setText(str(current_value))
        self.stateChanged.connect(self._state_changed)
        if tooltip:
            self.setToolTip(tooltip)
        self.alignment_flag = QtCore.Qt.AlignmentFlag.AlignCenter
        self.setChecked(value)

    def _state_changed(self):
        self.value = self.isChecked()

    def update_widget(self, value):
        pass


class ComboBox(WidgetData, QtWidgets.QComboBox):
    """QComboBox enhanced with WidgetData"""

    def __init__(
        self,
        value: str,
        range_: list,
        dtype: type | None = None,
        tooltip: str | None = None,
    ):
        WidgetData.__init__(self, value, range_, dtype)
        QtWidgets.QComboBox.__init__(self)
        self.addItems(range_)
        self.setCurrentText(value)
        self.currentIndexChanged.connect(self.__text_changed)
        if tooltip:
            self.setToolTip(tooltip)
        self.alignment_flag = None

    def __text_changed(self):
        self.value = self.currentText()

    def update_widget(self, value):
        value = self.value_to_string(value)
        if not value == self.currentText():
            self.setCurrentText(value)

    def value_check_range(self, value):
        if value in self.range:
            pass
        else:
            print("Warning: Selected item is not in list!")


class PushButton(WidgetData, QtWidgets.QPushButton):
    """
    QPushButton enhanced with WidgetData.

    Needs an additional callback function and button text.
    """

    def __init__(
        self,
        current_value: np.ndarray | str,
        dtype: type | None = None,
        button_function: Callable = None,
        button_text: str | None = None,
        tooltip: str | None = None,
    ):
        WidgetData.__init__(self, current_value, [], dtype)
        QtWidgets.QPushButton.__init__(self)
        self.value = current_value
        self.clicked.connect(lambda x: self.__button_clicked(button_function))
        if button_text:
            self.setText(button_text)
        if tooltip:
            self.setToolTip(tooltip)
        self.alignment_flag = QtCore.Qt.AlignmentFlag.AlignCenter

    def __button_clicked(self, button_function: Callable):
        self.value = button_function()

    def update_widget(self, value):
        pass
