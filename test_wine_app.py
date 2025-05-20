import pytest
import streamlit as st
from unittest.mock import patch, MagicMock
import json

# Mock secrets
@pytest.fixture(autouse=True)
def mock_secrets():
    with patch('streamlit.secrets', {
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
    with patch('streamlit.secrets', {
        "AZURE_ENDPOINT_URI": "https://mock-azure-endpoint.com",
        "AZURE_API_KEY": "mock-azure-key",
        "OPENAI_API_KEY": "mock-openai-key"
    }):
        import wine_app
        input_data = wine_app.input_data
        assert isinstance(input_data, dict)
        assert "columns" in input_data
        assert "data" in input_data
        assert len(input_data["columns"]) == 13
        assert len(input_data["data"][0]) == 13

def test_azure_ml_prediction_success(mock_streamlit):
    """Test successful Azure ML prediction flow"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [7.5]
    
    with patch('requests.post', return_value=mock_response) as mock_post, \
         patch('openai.OpenAI') as mock_openai:
        
        # Mock GPT response
        mock_gpt_response = MagicMock()
        mock_gpt_response.choices = [MagicMock()]
        mock_gpt_response.choices[0].message.content = "Test tasting note"
        mock_openai.return_value.chat.completions.create.return_value = mock_gpt_response
        
        # Import the module
        import wine_app
        
        # Simulate button click
        mock_streamlit['button'].return_value = True
        
        # Run the prediction code
        if mock_streamlit['button']():
            with mock_streamlit['spinner']():
                try:
                    body = json.dumps({"input_data": wine_app.input_data})
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {wine_app.AZURE_API_KEY}"
                    }
                    response = mock_post(wine_app.AZURE_ENDPOINT_URI, headers=headers, data=body)
                    
                    if response.status_code == 200:
                        result = response.json()
                        mock_streamlit['write']("✅ Raw model response:", result)
                        
                        quality = result[0] if isinstance(result, list) and len(result) > 0 else None
                        
                        if quality is not None:
                            mock_streamlit['success'](f"Predicted Wine Quality Score: {quality:.2f}")
                            
                            prompt = f"""
                            A {wine_app.color} wine has a predicted quality score of {quality:.1f}.
                            Characteristics:
                            - Alcohol: {wine_app.alcohol}%
                            - pH: {wine_app.ph}
                            - Residual sugar: {wine_app.residual_sugar}g/L
                            - Fixed acidity: {wine_app.fixed_acidity}
                            - Volatile acidity: {wine_app.volatile_acidity}
                            - Sulphates: {wine_app.sulphates}

                            Write a 2-sentence tasting description and suggest a food pairing.
                            """
                            
                            gpt_response = mock_openai.return_value.chat.completions.create(
                                model="gpt-4",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.7,
                                max_tokens=150
                            )
                            
                            mock_streamlit['markdown']("### 🍷 GPT-Generated Tasting Note")
                            mock_streamlit['write'](gpt_response.choices[0].message.content.strip())
                except Exception as e:
                    mock_streamlit['error'](f"⚠️ Unexpected error: {e}")
        
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
        # Import the module
        import wine_app
        
        # Simulate button click
        mock_streamlit['button'].return_value = True
        
        # Run the prediction code
        if mock_streamlit['button']():
            with mock_streamlit['spinner']():
                try:
                    body = json.dumps({"input_data": wine_app.input_data})
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {wine_app.AZURE_API_KEY}"
                    }
                    response = mock_post(wine_app.AZURE_ENDPOINT_URI, headers=headers, data=body)
                    
                    if response.status_code == 200:
                        # ... success handling ...
                        pass
                    else:
                        mock_streamlit['error'](f"❌ Azure ML call failed ({response.status_code})")
                        mock_streamlit['text'](response.text)
                except Exception as e:
                    mock_streamlit['error'](f"⚠️ Unexpected error: {e}")
        
        # Verify error message
        mock_streamlit['error'].assert_called_once()
        mock_streamlit['text'].assert_called_once_with("Internal Server Error")

def test_invalid_prediction_response(mock_streamlit):
    """Test handling of invalid prediction response"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []  # Empty response
    
    with patch('requests.post', return_value=mock_response) as mock_post:
        # Import the module
        import wine_app
        
        # Simulate button click
        mock_streamlit['button'].return_value = True
        
        # Run the prediction code
        if mock_streamlit['button']():
            with mock_streamlit['spinner']():
                try:
                    body = json.dumps({"input_data": wine_app.input_data})
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {wine_app.AZURE_API_KEY}"
                    }
                    response = mock_post(wine_app.AZURE_ENDPOINT_URI, headers=headers, data=body)
                    
                    if response.status_code == 200:
                        result = response.json()
                        mock_streamlit['write']("✅ Raw model response:", result)
                        
                        quality = result[0] if isinstance(result, list) and len(result) > 0 else None
                        
                        if quality is not None:
                            # ... success handling ...
                            pass
                        else:
                            mock_streamlit['warning']("Prediction returned no value.")
                except Exception as e:
                    mock_streamlit['error'](f"⚠️ Unexpected error: {e}")
        
        # Verify warning message
        mock_streamlit['warning'].assert_called_once_with("Prediction returned no value.")

def test_exception_handling(mock_streamlit):
    """Test exception handling in the main flow"""
    with patch('requests.post', side_effect=Exception("Test exception")) as mock_post:
        # Import the module
        import wine_app
        
        # Simulate button click
        mock_streamlit['button'].return_value = True
        
        # Run the prediction code
        if mock_streamlit['button']():
            with mock_streamlit['spinner']():
                try:
                    body = json.dumps({"input_data": wine_app.input_data})
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {wine_app.AZURE_API_KEY}"
                    }
                    response = mock_post(wine_app.AZURE_ENDPOINT_URI, headers=headers, data=body)
                except Exception as e:
                    mock_streamlit['error'](f"⚠️ Unexpected error: {e}")
        
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
        mock_streamlit['button'].return_value = True
        if mock_streamlit['button']():
            with mock_streamlit['spinner']():
                try:
                    body = json.dumps({"input_data": wine_app.input_data})
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {wine_app.AZURE_API_KEY}"
                    }
                    response = mock_post(wine_app.AZURE_ENDPOINT_URI, headers=headers, data=body)
                    if response.status_code == 200:
                        result = response.json()
                        mock_streamlit['write']("✅ Raw model response:", result)
                        quality = result[0] if isinstance(result, list) and len(result) > 0 else None
                        if quality is not None:
                            pass
                        else:
                            mock_streamlit['warning']("Prediction returned no value.")
                except Exception as e:
                    mock_streamlit['error'](f"⚠️ Unexpected error: {e}")
        mock_streamlit['warning'].assert_called_once_with("Prediction returned no value.")

def test_azure_ml_returns_list_with_none(mock_streamlit):
    """Test when Azure ML returns a list with None as the first element"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [None]
    
    with patch('requests.post', return_value=mock_response) as mock_post:
        import wine_app
        mock_streamlit['button'].return_value = True
        if mock_streamlit['button']():
            with mock_streamlit['spinner']():
                try:
                    body = json.dumps({"input_data": wine_app.input_data})
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {wine_app.AZURE_API_KEY}"
                    }
                    response = mock_post(wine_app.AZURE_ENDPOINT_URI, headers=headers, data=body)
                    if response.status_code == 200:
                        result = response.json()
                        mock_streamlit['write']("✅ Raw model response:", result)
                        quality = result[0] if isinstance(result, list) and len(result) > 0 else None
                        if quality is not None:
                            pass
                        else:
                            mock_streamlit['warning']("Prediction returned no value.")
                except Exception as e:
                    mock_streamlit['error'](f"⚠️ Unexpected error: {e}")
        mock_streamlit['warning'].assert_called_once_with("Prediction returned no value.")

def test_gpt4o_raises_exception(mock_streamlit):
    """Test when GPT-4o call raises an exception"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [7.5]
    
    with patch('requests.post', return_value=mock_response) as mock_post, \
         patch('openai.OpenAI') as mock_openai:
        mock_openai.return_value.chat.completions.create.side_effect = Exception("GPT error!")
        import wine_app
        mock_streamlit['button'].return_value = True
        if mock_streamlit['button']():
            with mock_streamlit['spinner']():
                try:
                    body = json.dumps({"input_data": wine_app.input_data})
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {wine_app.AZURE_API_KEY}"
                    }
                    response = mock_post(wine_app.AZURE_ENDPOINT_URI, headers=headers, data=body)
                    if response.status_code == 200:
                        result = response.json()
                        mock_streamlit['write']("✅ Raw model response:", result)
                        quality = result[0] if isinstance(result, list) and len(result) > 0 else None
                        if quality is not None:
                            try:
                                # This will raise
                                mock_openai.return_value.chat.completions.create()
                            except Exception as e:
                                mock_streamlit['error'](f"⚠️ Unexpected error: {e}")
                        else:
                            mock_streamlit['warning']("Prediction returned no value.")
                except Exception as e:
                    mock_streamlit['error'](f"⚠️ Unexpected error: {e}")
        mock_streamlit['error'].assert_any_call("⚠️ Unexpected error: GPT error!")

def test_azure_ml_returns_string(mock_streamlit):
    """Test when Azure ML returns a string instead of a list"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = "notalist"
    
    with patch('requests.post', return_value=mock_response) as mock_post:
        import wine_app
        mock_streamlit['button'].return_value = True
        if mock_streamlit['button']():
            with mock_streamlit['spinner']():
                try:
                    body = json.dumps({"input_data": wine_app.input_data})
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {wine_app.AZURE_API_KEY}"
                    }
                    response = mock_post(wine_app.AZURE_ENDPOINT_URI, headers=headers, data=body)
                    if response.status_code == 200:
                        result = response.json()
                        mock_streamlit['write']("✅ Raw model response:", result)
                        quality = result[0] if isinstance(result, list) and len(result) > 0 else None
                        if quality is not None:
                            pass
                        else:
                            mock_streamlit['warning']("Prediction returned no value.")
                except Exception as e:
                    mock_streamlit['error'](f"⚠️ Unexpected error: {e}")
        mock_streamlit['warning'].assert_called_once_with("Prediction returned no value.")

def test_predict_and_generate_note_success(mock_streamlit):
    """Test the refactored function for a successful prediction and GPT call."""
    from wine_app import predict_and_generate_note, input_data
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
         patch('openai.OpenAI') as mock_openai:
        # Set up the mock to return our prepared response
        mock_openai.return_value.chat.completions.create.return_value = mock_gpt_response
        
        # Debug: Print the mock setup
        print("\nMock setup:")
        print(f"mock_openai.return_value: {mock_openai.return_value}")
        print(f"mock_openai.return_value.chat: {mock_openai.return_value.chat}")
        print(f"mock_openai.return_value.chat.completions: {mock_openai.return_value.chat.completions}")
        print(f"mock_openai.return_value.chat.completions.create: {mock_openai.return_value.chat.completions.create}")
        
        predict_and_generate_note(
            input_data,
            "red", 10.0, 3.3, 1.9, 7.4, 0.7, 0.56
        )
        
        mock_streamlit['success'].assert_called_once()
        
        # Debug: Print all calls to st.write
        print("\nAll calls to st.write:")
        for call in mock_streamlit['write'].call_args_list:
            print(f"Call args: {call[0]}")
        
        # Debug: Print all calls to the GPT create method
        print("\nAll calls to GPT create:")
        for call in mock_openai.return_value.chat.completions.create.call_args_list:
            print(f"Call args: {call[0]}")
            print(f"Call kwargs: {call[1]}")
        
        # Check that st.write was called with a string containing 'Test tasting note'
        found = any(
            "Test tasting note" in str(call[0][0])
            for call in mock_streamlit['write'].call_args_list
        )
        assert found, "st.write was not called with the expected tasting note"

def test_predict_and_generate_note_warning(mock_streamlit):
    """Test the refactored function for a warning when prediction is None."""
    from wine_app import predict_and_generate_note, input_data
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    with patch('requests.post', return_value=mock_response):
        predict_and_generate_note(
            input_data,
            "red", 10.0, 3.3, 1.9, 7.4, 0.7, 0.56
        )
        mock_streamlit['warning'].assert_called_once_with("Prediction returned no value.")

def test_predict_and_generate_note_error(mock_streamlit):
    """Test the refactored function for an Azure ML error response."""
    from wine_app import predict_and_generate_note, input_data
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    with patch('requests.post', return_value=mock_response):
        predict_and_generate_note(
            input_data,
            "red", 10.0, 3.3, 1.9, 7.4, 0.7, 0.56
        )
        mock_streamlit['error'].assert_called_once()
        mock_streamlit['text'].assert_called_once_with("Internal Server Error")

def test_predict_and_generate_note_gpt_exception(mock_streamlit):
    """Test the refactored function for a GPT exception."""
    from wine_app import predict_and_generate_note, input_data
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [7.5]
    with patch('requests.post', return_value=mock_response), \
         patch('openai.OpenAI') as mock_openai:
        mock_openai.return_value.chat.completions.create.side_effect = Exception("GPT error!")
        predict_and_generate_note(
            input_data,
            "red", 10.0, 3.3, 1.9, 7.4, 0.7, 0.56
        )
        assert mock_streamlit['error'].call_count > 0 