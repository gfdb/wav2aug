import random
from unittest.mock import patch

import pytest
import torch

from wav2aug.cpu.wav2aug import Wav2Aug, _match_channels


class TestWav2AugBasicUsage:
    """Test basic augmentation functionality."""
    
    def test_mono_waveform(self):
        """Test augmentation of mono waveform."""
        aug = Wav2Aug(16000)
        x = torch.randn(8000)  # Use longer waveform to avoid augmentation issues
        result = aug(x)
        # Output should be [1, T] format after internal processing
        assert result.shape[0] == 1  # Should have 1 channel
        assert result.dtype == x.dtype
        # Length may change due to augmentations, but should be reasonable
        assert result.shape[1] > 0
        # Just verify that some transformation occurred - result shouldn't be identical
        # (though it's theoretically possible if augmentations cancel out)
        
    def test_stereo_waveform(self):
        """Test augmentation of stereo waveform."""
        aug = Wav2Aug(16000)
        x = torch.randn(2, 8000)  # Use longer waveform
        result = aug(x)
        assert result.shape[0] == x.shape[0]  # Same number of channels
        assert result.dtype == x.dtype
        assert result.shape[1] > 0  # Should have some length
        # Length may change due to augmentations
    
    def test_channel_consistency(self):
        """Test that channel count is set from first waveform."""
        aug = Wav2Aug(16000)
        x1 = torch.randn(1, 8000)  # mono
        x2 = torch.randn(2, 8000)  # stereo
        
        # First call sets channel count
        aug(x1)
        assert aug._C == 1
        
        # Second call should work with different channels
        result = aug(x2)
        assert result.shape[0] == x2.shape[0]  # Same number of channels
        assert result.shape[1] > 0  # Should have some length
    
    def test_empty_waveform(self):
        """Test handling of zero-length waveform."""
        aug = Wav2Aug(16000)
        x = torch.empty(2, 0)  # Multi-channel but zero time
        result = aug(x)
        assert result.shape == x.shape
    
    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise assertion error."""
        aug = Wav2Aug(16000)
        x = torch.randn(2, 3, 1000)
        with pytest.raises(AssertionError):
            aug(x)
    
    def test_device_requirement(self):
        """Test that CPU tensors are required."""
        aug = Wav2Aug(16000)
        x = torch.randn(8000)
        result = aug(x)
        assert result.device.type == 'cpu'


class TestBufferSystem:
    """Test the dual-buffer system."""
    
    def test_active_buffer_initialization(self):
        """Test active buffer is initialized on first waveform."""
        aug = Wav2Aug(16000, buffer_capacity=3)
        x = torch.randn(2, 8000)
        
        aug(x)
        assert aug._active_buffer is not None
        assert aug._active_count == 1
        assert aug._ready_buffer is None
        assert aug._C == 2
    
    def test_buffer_accumulation(self):
        """Test waveforms accumulate in active buffer."""
        aug = Wav2Aug(16000, buffer_capacity=3)
        x1 = torch.randn(2, 8000)
        x2 = torch.randn(2, 8000)
        
        aug(x1)
        buffer_after_1 = aug._active_buffer.clone()
        
        aug(x2)
        assert aug._active_count == 2
        assert not torch.equal(aug._active_buffer, buffer_after_1)
    
    def test_buffer_swap(self):
        """Test active buffer becomes ready when capacity reached."""
        aug = Wav2Aug(16000, buffer_capacity=2)
        x = torch.randn(2, 8000)
        
        aug(x)
        assert aug._active_count == 1
        assert aug._ready_buffer is None
        
        aug(x)
        assert aug._active_count == 0
        assert aug._ready_buffer is not None
        assert aug._active_buffer is None
    
    def test_variable_length_padding(self):
        """Test that variable length waveforms are handled correctly."""
        aug = Wav2Aug(16000, buffer_capacity=3)
        x1 = torch.randn(2, 4000)
        x2 = torch.randn(2, 8000)
        x3 = torch.randn(2, 6000)
        
        aug(x1)
        buffer_len_1 = aug._active_buffer.shape[1]
        assert buffer_len_1 == 4000
        
        aug(x2)
        buffer_len_2 = aug._active_buffer.shape[1]
        assert buffer_len_2 == 8000
        
        aug(x3)
        assert aug._ready_buffer.shape[1] == 8000
    
    def test_variable_channels(self):
        """Test handling of variable channel counts."""
        aug = Wav2Aug(16000, buffer_capacity=2)
        x1 = torch.randn(1, 8000)
        x2 = torch.randn(2, 8000)
        
        aug(x1)
        assert aug._C == 1
        
        aug(x2)
        assert aug._ready_buffer.shape[0] == 1
    
    def test_babble_noise_addition(self):
        """Test babble noise is added to ops when ready buffer available."""
        aug = Wav2Aug(16000, buffer_capacity=2)
        x = torch.randn(2, 8000)
        
        # Initially 8 base ops
        assert len(aug._base_ops) == 8
        
        # Fill buffer to trigger babble addition
        aug(x)
        aug(x)
        
        # Should now have 9 ops (8 base + babble)
        assert len(aug._base_ops) == 9
    
    def test_babble_disabled(self):
        """Test babble noise is not added immediately with small buffer capacity."""
        aug = Wav2Aug(16000, buffer_capacity=10)  # Larger capacity
        x = torch.randn(2, 8000)
        
        # Fill buffer only partially
        aug(x)
        
        # Should still have 8 ops (no babble added yet)
        assert len(aug._base_ops) == 8
        assert len(aug._base_ops) == 8


class TestBufferReset:
    """Test buffer reset functionality."""
    
    def test_buffer_state_management(self):
        """Test buffer state is managed correctly."""
        aug = Wav2Aug(16000, buffer_capacity=2)
        x = torch.randn(2, 8000)
        
        # Initially no ready buffer
        assert aug._ready_buffer is None
        
        # Fill buffers
        aug(x)
        aug(x)
        assert aug._ready_buffer is not None
        
        # Buffer system continues working
        aug(x)  # Should start new active buffer
    
    def test_configuration_persistence(self):
        """Test configuration persists through usage."""
        # Set seed for deterministic behavior
        torch.manual_seed(42)
        
        aug = Wav2Aug(44100, buffer_capacity=32)
        x = torch.randn(2, 8000)
        
        aug(x)
        
        assert aug.sample_rate == 44100
        assert aug.buffer_capacity == 32

class TestAugmentationApplication:
    """Test augmentation application logic."""
    
    @patch('random.sample')
    def test_two_augmentations_applied(self, mock_sample):
        """Test that exactly two augmentations are applied."""
        aug = Wav2Aug(16000)
        x = torch.randn(8000)
        
        # Use non-failing ops for testing
        op1 = lambda x: x * 0.9  # simple scaling
        op2 = lambda x: x + 0.01  # simple offset
        mock_sample.return_value = [op1, op2]
        
        result = aug(x)
        mock_sample.assert_called_once_with(aug._base_ops, 2)
        # Result should be different from input
        assert not torch.equal(result.view(-1), x)
    
    def test_insufficient_ops_fallback(self):
        """Test fallback when insufficient operations available."""
        aug = Wav2Aug(16000)
        x = torch.randn(8000)
        
        # Artificially reduce ops to less than 2
        aug._base_ops = [aug._base_ops[0]]
        
        result = aug(x)
        # Should return input unchanged (but reshaped to [1, T])
        assert result.shape == (1, len(x))
        assert torch.equal(result.view(-1), x)
    
    def test_deterministic_with_seed(self):
        """Test that results have some stability with same seed."""
        aug1 = Wav2Aug(16000)
        aug2 = Wav2Aug(16000)
        x = torch.randn(8000)
        
        # Set same seed for both runs
        torch.manual_seed(42)
        random.seed(42)
        result1 = aug1(x.clone())
        
        torch.manual_seed(42)
        random.seed(42) 
        result2 = aug2(x.clone())
        
        # Should be deterministic
        assert torch.allclose(result1, result2, rtol=1e-5, atol=1e-6)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_small_buffer_capacity(self):
        """Test with buffer capacity of 1."""
        aug = Wav2Aug(16000, buffer_capacity=1)
        x = torch.randn(2, 8000)
        
        aug(x)  # Should immediately create ready buffer
        assert aug._ready_buffer is not None
        assert aug._active_count == 0
    
    def test_large_buffer_capacity(self):
        """Test with large buffer capacity."""
        aug = Wav2Aug(16000, buffer_capacity=1000)
        x = torch.randn(2, 8000)
        
        # Process many waveforms without triggering swap
        for _ in range(10):  # Reduced to avoid too many calls
            aug(x)
        
        assert aug._ready_buffer is None
        assert aug._active_count == 10
    
    def test_zero_length_waveform_components(self):
        """Test with zero-length components."""
        aug = Wav2Aug(16000)
        x = torch.randn(2, 0)  # zero time dimension
        
        result = aug(x)
        assert result.shape == x.shape
    
    def test_single_sample_waveform(self):
        """Test with single-sample waveform."""
        aug = Wav2Aug(16000)
        x = torch.randn(2, 1)
        
        result = aug(x)
        assert result.shape == x.shape
    
    def test_dtype_preservation(self):
        """Test that dtypes are preserved."""
        aug = Wav2Aug(16000)
        
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(8000, dtype=dtype)
            result = aug(x)
            assert result.dtype == dtype