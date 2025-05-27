import pytest
import streamlit as st
from unittest.mock import patch, MagicMock
import json
import os

# Mock environment variables
@pytest.fixture(autouse=True)
def mock_env_vars():
    with patch.dict(os.environ, {
        "AZURE_ENDPOINT_URI": "https://mock-azure-endpoint.com",
        "AZURE_API_KEY": "mock-azure-key",
        "OPENAI_API_KEY": "mock-openai-key"
    }):
        yield

# Mock Streamlit
@pytest.fixture(autouse=True)
def mock_streamlit():
    with patch('streamlit.slider') as mock_slider, \
         patch('streamlit.selectbox') as mock_selectbox, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.spinner') as mock_spinner, \
         patch('streamlit.write') as mock_write, \
         patch('streamlit.success') as mock_success, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.warning') as mock_warning, \
         patch('streamlit.markdown') as mock_markdown, \
         patch('streamlit.text') as mock_text:
        
        # Configure mock return values
        mock_slider.return_value = 7.4
        mock_selectbox.side_effect = ["red", True]
        mock_button.return_value = True
        mock_spinner.return_value = MagicMock()
        
        yield {
            'slider': mock_slider,
            'selectbox': mock_selectbox,
            'button': mock_button,
            'spinner': mock_spinner,
            'write': mock_write,
            'success': mock_success,
            'error': mock_error,
            'warning': mock_warning,
            'markdown': mock_markdown,
            'text': mock_text
        }

def test_input_data_structure():
    """Test the structure of input_data dictionary"""
    # This test relies on the mock_streamlit fixture to set default values
    # for sliders and selectboxes, which are then used by wine_app global vars.
    with patch('streamlit.secrets', {
        "AZURE_ENDPOINT_URI": "https://mock-azure-endpoint.com",
        "AZURE_API_KEY": "mock-azure-key",
        "OPENAI_API_KEY": "mock-openai-key"
    }):
        import wine_app
        # Construct payload using the helper and current (mocked) global values from wine_app
        payload = wine_app.get_input_data_payload(
            wine_app.fixed_acidity, wine_app.volatile_acidity, wine_app.citric_acid,
            wine_app.residual_sugar, wine_app.chlorides, wine_app.free_sulfur_dioxide,
            wine_app.total_sulfur_dioxide, wine_app.density, wine_app.ph,
            wine_app.sulphates, wine_app.alcohol, wine_app.color, wine_app.qual_bool
        )
        assert isinstance(payload, dict)
        assert "columns" in payload
        assert "data" in payload
        assert len(payload["columns"]) == 13 # Assuming AZURE_ML_COLUMNS has 13 items
        assert len(payload["data"][0]) == 13

def test_azure_ml_prediction_success(mock_streamlit):
    """Test successful Azure ML prediction flow"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [7.5]
    
    # Mock GPT response parts
    mock_gpt_message = MagicMock()
    mock_gpt_message.content = "Test tasting note"
    mock_gpt_choice = MagicMock()
    mock_gpt_choice.message = mock_gpt_message
    mock_gpt_full_response = MagicMock()
    mock_gpt_full_response.choices = [mock_gpt_choice]

    with patch('requests.post', return_value=mock_response) as mock_post, \
         patch('openai.OpenAI') as mock_openai_class: # Patch the class
        
        mock_openai_instance = mock_openai_class.return_value
        mock_openai_instance.chat.completions.create.return_value = mock_gpt_full_response
        
        import wine_app # Import after mocks are set up

        # Prepare payload using mocked global values from wine_app
        payload = wine_app.get_input_data_payload(
            wine_app.fixed_acidity, wine_app.volatile_acidity, wine_app.citric_acid,
            wine_app.residual_sugar, wine_app.chlorides, wine_app.free_sulfur_dioxide,
            wine_app.total_sulfur_dioxide, wine_app.density, wine_app.ph,
            wine_app.sulphates, wine_app.alcohol, wine_app.color, wine_app.qual_bool
        )

        prompt_params = {
            "color": wine_app.color, "alcohol": wine_app.alcohol, "ph": wine_app.ph,
            "residual_sugar": wine_app.residual_sugar, "fixed_acidity": wine_app.fixed_acidity,
            "volatile_acidity": wine_app.volatile_acidity, "sulphates": wine_app.sulphates
        }

        # Call the main logic function
        # Note: function renamed to display_results and signature changed
        wine_app.display_results(
            azure_payload=payload,
            prompt_parameters=prompt_params
        )
        
        # Verify Azure ML call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['headers']['Authorization'] == 'Bearer mock-azure-key'
        
        # Verify success message
        mock_streamlit['success'].assert_called_once()
        mock_streamlit['markdown'].assert_called()

def test_azure_ml_prediction_failure(mock_streamlit):
    """Test Azure ML prediction failure handling"""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    
    with patch('requests.post', return_value=mock_response) as mock_post:
        import wine_app # Import after mocks are set up

        payload = wine_app.get_input_data_payload(
            wine_app.fixed_acidity, wine_app.volatile_acidity, wine_app.citric_acid,
            wine_app.residual_sugar, wine_app.chlorides, wine_app.free_sulfur_dioxide,
            wine_app.total_sulfur_dioxide, wine_app.density, wine_app.ph,
            wine_app.sulphates, wine_app.alcohol, wine_app.color, wine_app.qual_bool
        )

        prompt_params = {
            "color": wine_app.color, "alcohol": wine_app.alcohol, "ph": wine_app.ph,
            "residual_sugar": wine_app.residual_sugar, "fixed_acidity": wine_app.fixed_acidity,
            "volatile_acidity": wine_app.volatile_acidity, "sulphates": wine_app.sulphates
        }
        wine_app.display_results(
            azure_payload=payload,
            prompt_parameters=prompt_params
        )
        
        # Verify error message
        expected_error_msg = "❌ Azure ML call failed (500): Internal Server Error"
        mock_streamlit['error'].assert_called_once_with(expected_error_msg)
        mock_streamlit['text'].assert_not_called() # Error text is now part of st.error

def test_invalid_prediction_response(mock_streamlit):
    """Test handling of invalid prediction response"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []  # Empty response
    
    with patch('requests.post', return_value=mock_response) as mock_post:
        import wine_app

        payload = wine_app.get_input_data_payload(
            wine_app.fixed_acidity, wine_app.volatile_acidity, wine_app.citric_acid,
            wine_app.residual_sugar, wine_app.chlorides, wine_app.free_sulfur_dioxide,
            wine_app.total_sulfur_dioxide, wine_app.density, wine_app.ph,
            wine_app.sulphates, wine_app.alcohol, wine_app.color, wine_app.qual_bool
        )
        prompt_params = {
            "color": wine_app.color, "alcohol": wine_app.alcohol, "ph": wine_app.ph,
            "residual_sugar": wine_app.residual_sugar, "fixed_acidity": wine_app.fixed_acidity,
            "volatile_acidity": wine_app.volatile_acidity, "sulphates": wine_app.sulphates
        }
        wine_app.display_results(
            azure_payload=payload,
            prompt_parameters=prompt_params
        )
        
        # Verify warning message
        # Updated warning message for more specific scenarios
        mock_streamlit['warning'].assert_any_call("⚠️ Azure ML returned an empty list for prediction.")

def test_exception_handling(mock_streamlit):
    """Test exception handling in the main flow"""
    with patch('requests.post', side_effect=Exception("Test exception")) as mock_post:
        import wine_app

        payload = wine_app.get_input_data_payload(
            wine_app.fixed_acidity, wine_app.volatile_acidity, wine_app.citric_acid,
            wine_app.residual_sugar, wine_app.chlorides, wine_app.free_sulfur_dioxide,
            wine_app.total_sulfur_dioxide, wine_app.density, wine_app.ph,
            wine_app.sulphates, wine_app.alcohol, wine_app.color, wine_app.qual_bool
        )
        prompt_params = {
            "color": wine_app.color, "alcohol": wine_app.alcohol, "ph": wine_app.ph,
            "residual_sugar": wine_app.residual_sugar, "fixed_acidity": wine_app.fixed_acidity,
            "volatile_acidity": wine_app.volatile_acidity, "sulphates": wine_app.sulphates
        }
        wine_app.display_results(
            azure_payload=payload,
            prompt_parameters=prompt_params
        )
        
        # Verify error message
        mock_streamlit['error'].assert_called_once()
        assert "Test exception" in str(mock_streamlit['error'].call_args[0][0])

def test_azure_ml_returns_non_list(mock_streamlit):
    """Test when Azure ML returns a non-list (e.g., dict)"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"not": "a list"}
    
    with patch('requests.post', return_value=mock_response) as mock_post:
        import wine_app
        payload = wine_app.get_input_data_payload(
            wine_app.fixed_acidity, wine_app.volatile_acidity, wine_app.citric_acid,
            wine_app.residual_sugar, wine_app.chlorides, wine_app.free_sulfur_dioxide,
            wine_app.total_sulfur_dioxide, wine_app.density, wine_app.ph,
            wine_app.sulphates, wine_app.alcohol, wine_app.color, wine_app.qual_bool
        )
        prompt_params = {
            "color": wine_app.color, "alcohol": wine_app.alcohol, "ph": wine_app.ph,
            "residual_sugar": wine_app.residual_sugar, "fixed_acidity": wine_app.fixed_acidity,
            "volatile_acidity": wine_app.volatile_acidity, "sulphates": wine_app.sulphates
        }
        wine_app.display_results(
            azure_payload=payload,
            prompt_parameters=prompt_params
        )
        # Updated warning message
        mock_streamlit['warning'].assert_any_call("⚠️ Unexpected prediction result format from Azure ML: {'not': 'a list'}")

def test_azure_ml_returns_list_with_none(mock_streamlit):
    """Test when Azure ML returns a list with None as the first element."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [None]
    
    with patch('requests.post', return_value=mock_response) as mock_post:
        import wine_app
        mock_streamlit['button'].return_value = True
        payload = wine_app.get_input_data_payload(
            wine_app.fixed_acidity, wine_app.volatile_acidity, wine_app.citric_acid,
            wine_app.residual_sugar, wine_app.chlorides, wine_app.free_sulfur_dioxide,
            wine_app.total_sulfur_dioxide, wine_app.density, wine_app.ph,
            wine_app.sulphates, wine_app.alcohol, wine_app.color, wine_app.qual_bool
        )
        prompt_params = {
            "color": wine_app.color, "alcohol": wine_app.alcohol, "ph": wine_app.ph,
            "residual_sugar": wine_app.residual_sugar, "fixed_acidity": wine_app.fixed_acidity,
            "volatile_acidity": wine_app.volatile_acidity, "sulphates": wine_app.sulphates
        }
        wine_app.display_results(
            azure_payload=payload,
            prompt_parameters=prompt_params
        )
        # Updated warning message
        mock_streamlit['warning'].assert_any_call("⚠️ Prediction value from Azure ML is not a number: None")

def test_gpt4o_raises_exception(mock_streamlit):
    """Test when OpenAI call raises an exception"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [7.5]
    
    with patch('requests.post', return_value=mock_response) as mock_post, \
         patch('openai.OpenAI') as mock_openai:
        mock_openai.return_value.chat.completions.create.side_effect = Exception("GPT error!")
        
        import wine_app # Import after mocks
        payload = wine_app.get_input_data_payload(
            wine_app.fixed_acidity, wine_app.volatile_acidity, wine_app.citric_acid,
            wine_app.residual_sugar, wine_app.chlorides, wine_app.free_sulfur_dioxide,
            wine_app.total_sulfur_dioxide, wine_app.density, wine_app.ph,
            wine_app.sulphates, wine_app.alcohol, wine_app.color, wine_app.qual_bool
        )
        prompt_params = {
            "color": wine_app.color, "alcohol": wine_app.alcohol, "ph": wine_app.ph,
            "residual_sugar": wine_app.residual_sugar, "fixed_acidity": wine_app.fixed_acidity,
            "volatile_acidity": wine_app.volatile_acidity, "sulphates": wine_app.sulphates
        }
        wine_app.display_results(
            azure_payload=payload,
            prompt_parameters=prompt_params
        )
        mock_streamlit['error'].assert_any_call("⚠️ OpenAI Error: Error calling OpenAI: GPT error!")

def test_azure_ml_returns_string(mock_streamlit):
    """Test when Azure ML returns a string instead of a list"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = "notalist"
    
    with patch('requests.post', return_value=mock_response) as mock_post:
        import wine_app
        payload = wine_app.get_input_data_payload(
            wine_app.fixed_acidity, wine_app.volatile_acidity, wine_app.citric_acid,
            wine_app.residual_sugar, wine_app.chlorides, wine_app.free_sulfur_dioxide,
            wine_app.total_sulfur_dioxide, wine_app.density, wine_app.ph,
            wine_app.sulphates, wine_app.alcohol, wine_app.color, wine_app.qual_bool
        )
        prompt_params = {
            "color": wine_app.color, "alcohol": wine_app.alcohol, "ph": wine_app.ph,
            "residual_sugar": wine_app.residual_sugar, "fixed_acidity": wine_app.fixed_acidity,
            "volatile_acidity": wine_app.volatile_acidity, "sulphates": wine_app.sulphates
        }
        wine_app.display_results(
            azure_payload=payload,
            prompt_parameters=prompt_params
        )
        # Updated warning message
        mock_streamlit['warning'].assert_any_call("⚠️ Unexpected prediction result format from Azure ML: 'notalist'")

def test_display_prediction_and_notes_success(mock_streamlit):
    """Test the refactored function for a successful prediction and GPT call."""
    # Import after mocks are active via fixture
    import wine_app 
    from wine_app import display_results, get_input_data_payload # Renamed function

    # Use mocked global values from wine_app for default payload parts
    # These values are set by the mock_streamlit fixture for wine_app.fixed_acidity etc.
    payload = get_input_data_payload(
        wine_app.fixed_acidity, wine_app.volatile_acidity, wine_app.citric_acid, wine_app.residual_sugar,
        wine_app.chlorides, wine_app.free_sulfur_dioxide, wine_app.total_sulfur_dioxide, wine_app.density,
        wine_app.ph, wine_app.sulphates, wine_app.alcohol, wine_app.color, wine_app.qual_bool
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [7.5]
    
    # Create a mock message with content
    mock_message = MagicMock()
    mock_message.content = "Test tasting note"
    
    # Create a mock choice with the message
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    # Create a mock GPT response with the choice
    mock_gpt_response = MagicMock()
    mock_gpt_response.choices = [mock_choice]
    
    with patch('requests.post', return_value=mock_response) as mock_post, \
         patch('openai.OpenAI') as mock_openai_class: # Patch the class
        
        mock_openai_instance = mock_openai_class.return_value
        mock_openai_instance.chat.completions.create.return_value = mock_gpt_response
        
        prompt_parameters = {
            "color": "red",
            "alcohol": 10.0,
            "ph": 3.3,
            "residual_sugar": 1.9,
            "fixed_acidity": 7.4,
            "volatile_acidity": 0.7,
            "sulphates": 0.56
        }

        display_results( # Call the refactored function
            azure_payload=payload,
            prompt_parameters=prompt_parameters
        )
        
        mock_streamlit['success'].assert_called_once()
        
        # Debug: Print all calls to st.write
        print("\nAll calls to st.write:")
        for call in mock_streamlit['write'].call_args_list:
            print(f"Call args: {call[0]}")
        
        # Debug: Print all calls to the GPT create method
        print("\nAll calls to GPT create:")
        for call in mock_openai_instance.chat.completions.create.call_args_list:
            print(f"Call args: {call[0]}")
            print(f"Call kwargs: {call[1]}")
        
        # Check that st.write was called with a string containing 'Test tasting note'
        found = any(
            "Test tasting note" in str(call[0][0])
            for call in mock_streamlit['write'].call_args_list
        )
        assert found, "st.write was not called with the expected tasting note"

def test_display_prediction_and_notes_warning_on_no_prediction_value(mock_streamlit):
    """Test the refactored function for a warning when prediction is None."""
    import wine_app
    from wine_app import display_results, get_input_data_payload # Renamed function

    payload = get_input_data_payload(
        wine_app.fixed_acidity, wine_app.volatile_acidity, wine_app.citric_acid, wine_app.residual_sugar,
        wine_app.chlorides, wine_app.free_sulfur_dioxide, wine_app.total_sulfur_dioxide, wine_app.density,
        wine_app.ph, wine_app.sulphates, wine_app.alcohol, wine_app.color, wine_app.qual_bool
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    with patch('requests.post', return_value=mock_response):
        prompt_parameters = {
            "color": "red", "alcohol": 10.0, "ph": 3.3, "residual_sugar": 1.9,
            "fixed_acidity": 7.4, "volatile_acidity": 0.7, "sulphates": 0.56
        }
        display_results( # Call the refactored function
            azure_payload=payload,
            prompt_parameters=prompt_parameters
        )
        # Check for the more specific warning related to empty list
        mock_streamlit['warning'].assert_any_call("⚠️ Azure ML returned an empty list for prediction.")
        # Also check for the summary warning if quality is None
        mock_streamlit['warning'].assert_any_call("⚠️ Could not determine wine quality from Azure ML response. Cannot generate tasting notes.")


def test_display_prediction_and_notes_azure_error(mock_streamlit):
    """Test the refactored function for an Azure ML error response."""
    import wine_app
    from wine_app import display_results, get_input_data_payload # Renamed function

    payload = get_input_data_payload(
        wine_app.fixed_acidity, wine_app.volatile_acidity, wine_app.citric_acid, wine_app.residual_sugar,
        wine_app.chlorides, wine_app.free_sulfur_dioxide, wine_app.total_sulfur_dioxide, wine_app.density,
        wine_app.ph, wine_app.sulphates, wine_app.alcohol, wine_app.color, wine_app.qual_bool
    )

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    with patch('requests.post', return_value=mock_response):
        prompt_parameters = {
            "color": "red", "alcohol": 10.0, "ph": 3.3, "residual_sugar": 1.9,
            "fixed_acidity": 7.4, "volatile_acidity": 0.7, "sulphates": 0.56
        }
        display_results( # Call the refactored function
            azure_payload=payload,
            prompt_parameters=prompt_parameters
        )
        # The error message is now formatted by fetch_prediction_from_azure
        expected_error_msg = "❌ Azure ML call failed (500): Internal Server Error"
        mock_streamlit['error'].assert_called_once_with(expected_error_msg)
        # st.text is no longer directly called for this; error message is consolidated
        mock_streamlit['text'].assert_not_called()


def test_display_prediction_and_notes_gpt_exception(mock_streamlit):
    """Test the refactored function for a GPT exception."""
    import wine_app
    from wine_app import display_results, get_input_data_payload # Renamed function

    payload = get_input_data_payload(
        wine_app.fixed_acidity, wine_app.volatile_acidity, wine_app.citric_acid, wine_app.residual_sugar,
        wine_app.chlorides, wine_app.free_sulfur_dioxide, wine_app.total_sulfur_dioxide, wine_app.density,
        wine_app.ph, wine_app.sulphates, wine_app.alcohol, wine_app.color, wine_app.qual_bool
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [7.5]
    with patch('requests.post', return_value=mock_response), \
         patch('openai.OpenAI') as mock_openai_class:
        mock_openai_instance = mock_openai_class.return_value
        mock_openai_instance.chat.completions.create.side_effect = Exception("GPT error!")
        
        prompt_parameters = {
            "color": "red", "alcohol": 10.0, "ph": 3.3, "residual_sugar": 1.9,
            "fixed_acidity": 7.4, "volatile_acidity": 0.7, "sulphates": 0.56
        }
        display_results( # Call the refactored function
            azure_payload=payload,
            prompt_parameters=prompt_parameters
        )
        mock_streamlit['error'].assert_called_once_with("⚠️ OpenAI Error: Error calling OpenAI: GPT error!")