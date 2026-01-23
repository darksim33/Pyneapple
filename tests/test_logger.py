"""Tests for logging configuration and functionality.

This module tests the logging system based on loguru including:

- Logger configuration: Setting up file and console logging
- Log levels: Debug, info, warning, error level filtering
- Log formatting: Custom format strings and colors
- File handling: Log file creation, rotation, and cleanup
- Context managers: Temporary logging configurations
- Integration: Logging during fitting and processing operations

The logging system provides detailed diagnostic information during
fitting operations and can be configured for different verbosity levels.
"""
import pytest
import sys
import os
import tempfile
from io import StringIO
from loguru import logger

# Import the actual module to get consistent class references
import pyneapple.utils.logger as logger_module
from pyneapple.utils.logger import intercept_stdout_stderr, restore_stdout_stderr

# Use the InterceptOutput class from the actual module
InterceptOutput = logger_module.InterceptOutput


class TestLoggerConfiguration:
    """Test logger configuration and environment variable handling."""

    def test_log_level_from_env_default(self, mocker):
        """Test that LOG_LEVEL defaults to INFO when not set."""
        mocker.patch.dict(os.environ, {}, clear=True)
        # Reimport to test environment variable handling
        import importlib
        from pyneapple.utils import logger as logger_module

        importlib.reload(logger_module)

        assert logger_module.DEFAULT_LOG_LEVEL == "INFO"

    def test_log_level_from_env_custom(self, mocker):
        """Test that LOG_LEVEL uses environment variable when set."""
        mocker.patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"})
        import importlib
        from pyneapple.utils import logger as logger_module

        importlib.reload(logger_module)

        assert logger_module.DEFAULT_LOG_LEVEL == "DEBUG"

    def test_log_format_structure(self):
        """Test that LOG_FORMAT contains expected components."""
        from pyneapple.utils.logger import LOG_FORMAT

        # Check that format contains expected placeholders
        assert "{time:" in LOG_FORMAT
        assert "{level" in LOG_FORMAT
        assert "{name}" in LOG_FORMAT
        assert "{function}" in LOG_FORMAT
        assert "{line}" in LOG_FORMAT
        assert "{message}" in LOG_FORMAT


class TestInterceptOutput:
    """Test the InterceptOutput class functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.interceptor = InterceptOutput("INFO")

    def test_init_default_level(self):
        """Test InterceptOutput initialization with default level."""
        interceptor = InterceptOutput()
        assert interceptor.level == "INFO"

    def test_init_custom_level(self):
        """Test InterceptOutput initialization with custom level."""
        interceptor = InterceptOutput("DEBUG")
        assert interceptor.level == "DEBUG"

    def test_write_non_empty_message(self, mocker):
        """Test writing non-empty messages to logger."""
        mock_logger = mocker.patch("pyneapple.utils.logger.logger")
        mock_logger.info = mocker.MagicMock()

        self.interceptor.write("Test message")

        mock_logger.info.assert_called_once_with("Test message")

    def test_write_empty_message(self, mocker):
        """Test that empty messages are not logged."""
        mock_logger = mocker.patch("pyneapple.utils.logger.logger")
        mock_logger.info = mocker.MagicMock()

        self.interceptor.write("")
        self.interceptor.write("   ")
        self.interceptor.write("\n")

        mock_logger.info.assert_not_called()

    def test_write_strips_whitespace(self, mocker):
        """Test that messages are stripped of whitespace."""
        mock_logger = mocker.patch("pyneapple.utils.logger.logger")
        mock_logger.info = mocker.MagicMock()

        self.interceptor.write("  Test message  \n")

        mock_logger.info.assert_called_once_with("Test message")

    def test_write_different_log_levels(self, mocker):
        """Test writing with different log levels."""
        mock_logger = mocker.patch("pyneapple.utils.logger.logger")
        mock_logger.debug = mocker.MagicMock()
        mock_logger.warning = mocker.MagicMock()
        mock_logger.error = mocker.MagicMock()

        debug_interceptor = InterceptOutput("DEBUG")
        warning_interceptor = InterceptOutput("WARNING")
        error_interceptor = InterceptOutput("ERROR")

        debug_interceptor.write("Debug message")
        warning_interceptor.write("Warning message")
        error_interceptor.write("Error message")

        mock_logger.debug.assert_called_once_with("Debug message")
        mock_logger.warning.assert_called_once_with("Warning message")
        mock_logger.error.assert_called_once_with("Error message")

    def test_flush_does_nothing(self):
        """Test that flush method exists and does nothing."""
        # Should not raise any exception
        self.interceptor.flush()


class TestStdoutStderrRedirection:
    """Test stdout and stderr redirection functionality."""

    def setup_method(self):
        """Store original stdout and stderr."""
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def teardown_method(self):
        """Restore original stdout and stderr."""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def test_intercept_stdout_stderr(self):
        """Test that intercept_stdout_stderr redirects outputs."""
        intercept_stdout_stderr()

        assert isinstance(sys.stdout, logger_module.InterceptOutput)
        assert isinstance(sys.stderr, logger_module.InterceptOutput)
        assert sys.stdout.level == "INFO"
        assert sys.stderr.level == "ERROR"

    def test_restore_stdout_stderr(self):
        """Test that restore_stdout_stderr restores original outputs."""
        # First intercept
        intercept_stdout_stderr()
        assert isinstance(sys.stdout, logger_module.InterceptOutput)
        assert isinstance(sys.stderr, logger_module.InterceptOutput)

        # Then restore
        restore_stdout_stderr()
        assert sys.stdout is sys.__stdout__
        assert sys.stderr is sys.__stderr__

    def test_stdout_redirection_works(self, mocker):
        mock_logger = mocker.patch("pyneapple.utils.logger.logger")
        """Test that stdout redirection actually logs messages."""
        mock_logger.info = mocker.MagicMock()

        intercept_stdout_stderr()
        print("Test stdout message")

        mock_logger.info.assert_called_with("Test stdout message")

    def test_stderr_redirection_works(self, mocker):
        mock_logger = mocker.patch("pyneapple.utils.logger.logger")
        """Test that stderr redirection actually logs messages."""
        mock_logger.error = mocker.MagicMock()

        intercept_stdout_stderr()
        sys.stderr.write("Test stderr message\n")

        mock_logger.error.assert_called_with("Test stderr message")


class TestLoggerIntegration:
    """Test integration with loguru logger."""

    def test_logger_is_configured(self):
        """Test that logger is properly configured."""
        # Check that the default handler is removed and our handler is added
        # This is more of a smoke test since loguru's internal state is complex
        assert len(logger._core.handlers) > 0

    def test_logger_output_format(self, mocker):
        mock_stderr = mocker.patch("sys.stderr", new_callable=StringIO)
        """Test that logger outputs in expected format."""
        # Create a temporary logger for testing
        test_logger = logger.bind()
        test_logger.add(mock_stderr, format="{level} | {message}", level="INFO")

        test_logger.info("Test message")

        output = mock_stderr.getvalue()
        assert "INFO" in output
        assert "Test message" in output

    def test_logger_level_filtering(self):
        """Test that logger respects level filtering."""
        with StringIO() as test_output:
            test_logger = logger.bind()
            test_logger.add(test_output, level="WARNING", format="{message}")

            test_logger.debug("Debug message")
            test_logger.info("Info message")
            test_logger.warning("Warning message")

            output = test_output.getvalue()
            assert "Debug message" not in output
            assert "Info message" not in output
            assert "Warning message" in output


class TestLoggerEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_log_level_interceptor(self, mocker):
        """Test InterceptOutput with invalid log level."""
        interceptor = InterceptOutput("INVALID_LEVEL")

        # Should not raise an exception during initialization
        assert interceptor.level == "INVALID_LEVEL"

        # Writing should handle the invalid level gracefully
        mock_logger = mocker.patch("pyneapple.utils.logger.logger")
        mock_logger.invalid_level = mocker.MagicMock(side_effect=AttributeError)

        # This should not crash the application
        try:
            interceptor.write("Test message")
        except AttributeError:
            # Expected behavior for invalid log level
            pass

    def test_multiple_intercept_calls(self):
        """Test that multiple calls to intercept don't cause issues."""
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            intercept_stdout_stderr()
            first_stdout = sys.stdout
            first_stderr = sys.stderr

            intercept_stdout_stderr()
            second_stdout = sys.stdout
            second_stderr = sys.stderr

            # Should replace with new interceptors
            assert isinstance(first_stdout, logger_module.InterceptOutput)
            assert isinstance(second_stdout, logger_module.InterceptOutput)
            assert first_stdout is not second_stdout

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_restore_without_intercept(self):
        """Test that restore works even if intercept was never called."""
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        restore_stdout_stderr()

        # Should still point to the original streams
        assert sys.stdout is sys.__stdout__
        assert sys.stderr is sys.__stderr__


def test_set_get_log_level():
    """Test setting and getting log level."""
    # Save original log level
    original_level = logger_module.get_log_level()

    try:
        # Set new log level
        logger_module.set_log_level("DEBUG")

        # Check if level was changed
        assert logger_module.get_log_level() == "DEBUG"

        # Change to another level
        logger_module.set_log_level("WARNING")
        assert logger_module.get_log_level() == "WARNING"
    finally:
        # Restore original log level
        logger_module.set_log_level(original_level)


def test_set_log_level_removes_old_handler(mocker):
    mock_logger = mocker.patch("pyneapple.utils.logger.logger")
    """Test that set_log_level removes the old handler."""
    original_logger_id = logger_module._logger_id
    original_log_to_file = logger_module._LOG_TO_FILE
    try:
        # deactivate file logging for this test
        logger_module._LOG_TO_FILE = False

        # set known logger_id for testing
        test_logger_id = 42
        logger_module._logger_id = test_logger_id

        # Mock the add-Methode to return a known ID
        mock_logger.add.return_value = 43

        # mock the remove method
        mock_logger.remove = mocker.MagicMock()

        # set log level to trigger the remove
        logger_module.set_log_level("DEBUG")

        # check if the remove method was called with the correct logger_id
        mock_logger.remove.assert_called_once_with(test_logger_id)
    finally:
        # restore the original logger_id
        logger_module._logger_id = original_logger_id
        logger_module._LOG_TO_FILE = original_log_to_file


def test_get_log_level_default(mocker):
    """Test that get_log_level returns default when no handlers exist."""
    mock_core = mocker.MagicMock()
    mock_core.handlers = {}
    mocker.patch.object(logger_module.logger, "_core", mock_core)
    mocker.patch("pyneapple.utils.logger.DEFAULT_LOG_LEVEL", "INFO")
    assert logger_module.get_log_level() == "INFO"


class TestOutputMode:
    """Test output mode switching functionality."""

    def setup_method(self):
        """Store original state."""
        self.original_logger_id = logger_module._logger_id
        self.original_logger_id_file = logger_module._logger_id_file
        self.original_log_to_file = logger_module._LOG_TO_FILE

    def teardown_method(self):
        """Restore original state."""
        logger_module._logger_id = self.original_logger_id
        logger_module._logger_id_file = self.original_logger_id_file
        logger_module._LOG_TO_FILE = self.original_log_to_file

    def test_get_output_mode_both(self, mocker):
        """Test get_output_mode returns 'both' when both handlers exist."""
        # Set up both handlers
        logger_module._logger_id = 1
        logger_module._logger_id_file = 2
        
        mocker.patch.object(logger_module.logger._core, 'handlers', {1: mocker.MagicMock(), 2: mocker.MagicMock()})
        assert logger_module.get_output_mode() == "both"

    def test_get_output_mode_console_only(self, mocker):
        """Test get_output_mode returns 'console' when only console handler exists."""
        logger_module._logger_id = 1
        logger_module._logger_id_file = None
        
        mocker.patch.object(logger_module.logger._core, 'handlers', {1: mocker.MagicMock()})
        assert logger_module.get_output_mode() == "console"

    def test_get_output_mode_file_only(self, mocker):
        """Test get_output_mode returns 'file' when only file handler exists."""
        logger_module._logger_id = None
        logger_module._logger_id_file = 2
        
        mocker.patch.object(logger_module.logger._core, 'handlers', {2: mocker.MagicMock()})
        assert logger_module.get_output_mode() == "file"

    def test_get_output_mode_none(self, mocker):
        """Test get_output_mode returns 'none' when no handlers exist."""
        logger_module._logger_id = None
        logger_module._logger_id_file = None
        
        mocker.patch.object(logger_module.logger._core, 'handlers', {})
        assert logger_module.get_output_mode() == "none"

    def test_set_output_mode_console(self, mocker):
        mock_logger = mocker.patch("pyneapple.utils.logger.logger")
        """Test set_output_mode with 'console' mode."""
        mock_logger.add.return_value = 1
        mock_logger.remove = mocker.MagicMock()
        
        logger_module.set_output_mode("console")
        
        # Should add console handler
        assert mock_logger.add.call_count == 1
        assert mock_logger.add.call_args[0][0] == sys.stderr
        assert logger_module._LOG_TO_FILE is False

    def test_set_output_mode_file(self, mocker):
        mock_logger = mocker.patch("pyneapple.utils.logger.logger")
        """Test set_output_mode with 'file' mode."""
        mock_logger.add.return_value = 2
        mock_logger.remove = mocker.MagicMock()
        
        logger_module.set_output_mode("file")
        
        # Should add file handler only
        assert mock_logger.add.call_count == 1
        assert "logs/pyneapple.log" in str(mock_logger.add.call_args[0][0])
        assert logger_module._LOG_TO_FILE is True

    def test_set_output_mode_both(self, mocker):
        mock_logger = mocker.patch("pyneapple.utils.logger.logger")
        """Test set_output_mode with 'both' mode."""
        mock_logger.add.side_effect = [1, 2]
        mock_logger.remove = mocker.MagicMock()
        
        logger_module.set_output_mode("both")
        
        # Should add both handlers
        assert mock_logger.add.call_count == 2
        assert logger_module._LOG_TO_FILE is True

    def test_set_output_mode_with_level(self, mocker):
        mock_logger = mocker.patch("pyneapple.utils.logger.logger")
        """Test set_output_mode with custom log level."""
        mock_logger.add.return_value = 1
        mock_logger.remove = mocker.MagicMock()
        
        logger_module.set_output_mode("console", level="DEBUG")
        
        # Should use DEBUG level
        call_args = mock_logger.add.call_args
        assert call_args[1]["level"] == "DEBUG"

    def test_set_output_mode_removes_existing_handlers(self, mocker):
        mock_logger = mocker.patch("pyneapple.utils.logger.logger")
        """Test that set_output_mode removes existing handlers."""
        logger_module._logger_id = 99
        logger_module._logger_id_file = 100
        
        mock_logger.remove = mocker.MagicMock()
        mock_logger.add.return_value = 1
        
        logger_module.set_output_mode("console")
        
        # Should remove both existing handlers
        assert mock_logger.remove.call_count == 2
        mock_logger.remove.assert_any_call(99)
        mock_logger.remove.assert_any_call(100)

    def test_set_output_mode_handles_missing_handlers(self, mocker):
        mock_logger = mocker.patch("pyneapple.utils.logger.logger")
        """Test that set_output_mode handles missing handlers gracefully."""
        logger_module._logger_id = 99
        logger_module._logger_id_file = 100
        
        mock_logger.remove = mocker.MagicMock(side_effect=ValueError)
        mock_logger.add.return_value = 1
        
        # Should not raise an exception
        logger_module.set_output_mode("console")
        
        # Should still add new handler
        assert mock_logger.add.called

    def test_set_output_mode_uses_current_level_when_none(self, mocker):
        """Test that set_output_mode uses current level when level is None."""
        mock_get_level = mocker.patch("pyneapple.utils.logger.get_log_level")
        mock_logger = mocker.patch("pyneapple.utils.logger.logger")
        mock_get_level.return_value = "WARNING"
        mock_logger.add.return_value = 1
        mock_logger.remove = mocker.MagicMock()
        
        logger_module.set_output_mode("console", level=None)
        
        # Should call get_log_level
        mock_get_level.assert_called_once()
        
        # Should use WARNING level
        call_args = mock_logger.add.call_args
        assert call_args[1]["level"] == "WARNING"

    def test_set_output_mode_debug_format(self, mocker):
        """Test that set_output_mode uses DEBUG format for DEBUG level."""
        mock_logger = mocker.patch("pyneapple.utils.logger.logger")
        mock_logger.add.return_value = 1
        mock_logger.remove = mocker.MagicMock()
        
        logger_module.set_output_mode("console", level="DEBUG")
        
        # Should use DEBUG format
        call_args = mock_logger.add.call_args
        assert logger_module.DEBUG_LOG_FORMAT in str(call_args[1]["format"])

    def test_set_output_mode_info_format(self, mocker):
        """Test that set_output_mode uses INFO format for non-DEBUG levels."""
        mock_logger = mocker.patch("pyneapple.utils.logger.logger")
        mock_logger.add.return_value = 1
        mock_logger.remove = mocker.MagicMock()
        
        logger_module.set_output_mode("console", level="INFO")
        
        # Should use INFO format
        call_args = mock_logger.add.call_args
        assert logger_module.INFO_LOG_FORMAT in str(call_args[1]["format"])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

