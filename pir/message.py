from enum import IntEnum
from .encoding import decode_little_endian, encode_little_endian

class PIRScheme(IntEnum):
    NAIVE = 1
    SQRT = 2
    OPTIMIZED_SQRT = 3

class PIRMessageType(IntEnum):
    QUERY = 1
    ANSWER = 2
    HINT = 3

class PIRMessage:
    """
    This class represents a message in the PIR protocol. It contains the type of the message, the PIR scheme, and the payload.
    """
    def __init__(self, msg_type: PIRMessageType, scheme: PIRScheme, payload: bytes):
        self.msg_type = msg_type
        self.scheme = scheme
        self.payload = payload

    def to_bytes(self) -> bytes:
        """
        This method encodes the PIRMessage into bytes for transmission.
        msg_type | scheme | payload
        """
        return encode_little_endian(self.msg_type.value, 1) + encode_little_endian(self.scheme.value, 1) + self.payload
    
    @classmethod
    def from_bytes(cls, data: bytes):
        """
        This method decodes bytes into a PIRMessage object.
        """
        assert len(data) >= 2, "Data must be at least 2 bytes long to decode msg_type and scheme."

        msg_type_value = decode_little_endian(data[0:1])
        try:
            msg_type = PIRMessageType(msg_type_value)
        except ValueError:
            raise ValueError(f"Invalid message type value: {msg_type_value}. Must be one of {list(PIRMessageType)}")
        
        scheme = PIRScheme(decode_little_endian(data[1:2]))
        payload = data[2:]

        return cls(msg_type, scheme, payload)
    
    def __repr__(self):
        return f"PIRMessage(type={self.msg_type}, scheme={self.scheme}, payload_length={len(self.payload)})"