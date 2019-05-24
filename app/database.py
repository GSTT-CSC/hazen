# -*- coding: utf-8 -*-

"""Database module, including the SQLAlchemy database object and DB-related utilities."""

import uuid
from typing import Sequence, Tuple, Union

import sqlalchemy.types as types
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import relationship

from app import db
from sqlalchemy_utils.types.arrow import ArrowType
#
# from backend.extensions import db
from app.util.utils import base62_decode, base62_encode

# Alias common SQLAlchemy names
Column = db.Column
relationship = relationship
UUID = postgresql.UUID
JSONB = postgresql.JSONB


class CRUDMixin(object):
    """Mixin that adds convenience methods for CRUD (create, read, update, delete) operations."""

    @classmethod
    def create(cls, **kwargs):
        """Create a new record and save it the database."""
        instance = cls(**kwargs)
        return instance.save()

    def update(self, commit=True, **kwargs):
        """Update specific fields of a record."""
        for attr, value in kwargs.items():
            setattr(self, attr, value)
        return commit and self.save() or self

    def save(self, commit=True):
        """Save the record."""
        db.session.add(self)
        if commit:
            db.session.commit()
        return self

    def delete(self, commit=True):
        """Remove the record from the database."""
        db.session.delete(self)
        return commit and db.session.commit()


class Model(CRUDMixin, db.Model):
    """Base model class that includes CRUD convenience methods."""

    __abstract__ = True

    def __getitem__(self, item):
        """Convenience method that shortcuts `obj['foo']` to `obj.foo`."""
        return getattr(self, item)


# From Mike Bayer's "Building the app" talk
# https://speakerdeck.com/zzzeek/building-the-app
class SurrogatePK(object):
    """A mixin that adds a surrogate uuid 'primary key' column named ``id`` to any declarative-mapped class."""

    __table_args__ = {'extend_existing': True}

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    @classmethod
    def get_by_id(cls, record_id: Union[uuid.UUID, str]):
        """Get record by UUID."""
        if isinstance(record_id, str):
            try:
                record_id = uuid.UUID(record_id)
            except ValueError:
                return None
        return cls.query.get(record_id)

    @classmethod
    def get_by_id_or_hash(cls, identifier: Union[uuid.UUID, str]):
        """Get record by UUID or base62-hash."""

        if isinstance(identifier, str):
            try:
                # first, attempt to convert literal UUID strings
                identifier = uuid.UUID(identifier)
            except ValueError:
                # second, assume it is a hash and base62-decode it into a UUID instead
                try:
                    decoded = base62_decode(identifier)
                    while decoded > 2**128:
                        # random hash is too long and needs to be truncated to fit into UUID
                        identifier = identifier[:-1]
                        decoded = base62_decode(identifier)
                except ValueError:
                    # abort early if the identifier cannot be decoded as hash
                    return None

                try:
                    identifier = uuid.UUID(int=decoded)
                except ValueError:
                    return None

        if not isinstance(identifier, uuid.UUID):
            # unable to determine the UUID altogether
            return None

        return cls.get_by_id(identifier)

    @property
    def hash(self) -> str:
        """Hash generated from primary key."""
        if self.id:
            return base62_encode(self.id.int)
        return None

    def __repr__(self):
        """Represent instance as a recognisable string."""
        return '<{}({:.6})>'.format(self.__class__.__name__, self.hash)


def reference_col(tablename, nullable=False, pk_name='id', **kwargs):
    """Column that adds primary key foreign key reference.

    Usage: ::

        category_id = reference_col('category')
        category = relationship('Category', backref='categories')
    """
    return db.Column(db.ForeignKey('{0}.{1}'.format(tablename, pk_name)), nullable=nullable, **kwargs)


class CreatedTimestampMixin:
    """Add an `created_at` column that automatically populates at creation."""

    created_at = Column(ArrowType, nullable=False, default=db.func.now())


# noinspection PyClassHasNoInit
class Cube(types.UserDefinedType):
    """Support for PostgreSQL's cube extension."""

    @property
    def python_type(self):
        """Python equivalent type."""
        return Tuple[float]

    # noinspection PyMethodMayBeStatic
    def get_col_spec(self):
        """PostgreSQL column type specification."""
        return 'cube'

    def bind_processor(self, dialect):
        """Convert python type to PostgreSQL."""
        def process(values: Sequence[float]) -> str:
            if values:
                return ','.join(map(str, values))
            return None
        return process

    def result_processor(self, dialect, coltype):
        """Convert PostgreSQL type to python."""
        def process(value: str) -> Tuple:
            if value:
                return tuple(map(lambda x: float(x.strip()), value.lstrip('(').rstrip(')').split(',')))
            return ()
        return process
