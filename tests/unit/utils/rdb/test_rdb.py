import logging
import pytest

from app.config import Config  
from app.utils.rdb import RelationDB


logger = logging.getLogger(__name__)


@pytest.fixture
def relation_db():
    """Fixture for the RelationDB class."""

    config = Config()
    return RelationDB(config)


def test_relation_db_execute(relation_db):
    """Test the execute function of the RelationDB class."""

    query = "SELECT name FROM sys.tables"

    result = relation_db.execute(query)

    assert len(result) > 0