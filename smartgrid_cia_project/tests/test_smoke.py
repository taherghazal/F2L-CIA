from src.data_simulator import generate_smart_meter_data
from src.crypto_utils import encrypt_packet, verify_and_decrypt


def test_pipeline_smoke():
    df = generate_smart_meter_data(num_meters=1, seq_len=10)
    pkt = encrypt_packet(df.iloc[0].to_dict())
    decoded, verified = verify_and_decrypt(pkt)
    assert verified == 1
    assert int(decoded['meter_id']) == int(df.iloc[0]['meter_id'])
