from app.utils.conversation import Source


def test_source_to_json():
    """This function tests source to json function."""
    
    source = Source(source_urls={1: 'https://test.net/test/test.txt', 2: 'sample/test.txt'})

    assert source.to_json() == {'source_urls': [{"index": "[1]", "file_name": "test.txt", "url": "https://test.net/test/test.txt"}, 
                                                {"index": "[2]", "file_name": "test.txt", "url": "file://sample/test.txt"}]}