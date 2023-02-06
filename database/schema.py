from typing import List
from sqlalchemy import ForeignKey
from sqlalchemy import Column, Table
from sqlalchemy import String, LargeBinary, Boolean
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

class Base(DeclarativeBase):
    pass

operation_gestures = Table(
    "operation_gestures",
    Base.metadata,
    Column("operation_id", ForeignKey("operations.id")),
    Column("gesture_id", ForeignKey("gestures.id")),
)

class Gesture(Base):
    __tablename__ = "gestures"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    trained: Mapped[bool] = mapped_column(Boolean())

    # 一个手势有多个数据集记录
    dataset: Mapped[List["Data"]] = relationship(
        back_populates="gesture", cascade="all, delete-orphan"
    )

    # 包含当前手势的操作列表
    operations: Mapped[List["Operation"]] = relationship(
        secondary=operation_gestures, back_populates="gestures"
    )

    def __repr__(self) -> str:
        return f"Dataset(id={self.id!r}, name={self.name!r})"


class Data(Base):
    __tablename__ = "data_set"

    id: Mapped[int] = mapped_column(primary_key=True)
    gesture_id: Mapped[int] = mapped_column(ForeignKey("gestures.id"))
    data: Mapped[bytes] = mapped_column(LargeBinary())

    # 多个数据集记录对应一个手势
    gesture: Mapped["Gesture"] = relationship(back_populates="dataset")

    def __repr__(self) -> str:
        return f"Dataset(id={self.id!r}, gesture_id={self.gesture_id!r})"


class OperationType(Base):
    __tablename__ = "operation_types"

    id: Mapped[int] = mapped_column(primary_key=True)
    type_name: Mapped[str] = mapped_column(String(30))

    # 一个操作类型对应多个操作实例
    operations: Mapped[List["Operation"]] = relationship(
        back_populates="operation_type", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"OperationType(id={self.id!r}, type_name={self.type_name!r})"


class Operation(Base):
    __tablename__ = "operations"

    id: Mapped[int] = mapped_column(primary_key=True)
    type_id: Mapped[int] = mapped_column(ForeignKey("operation_types.id"))
    name: Mapped[str] = mapped_column(String(30))
    extra_data: Mapped[str] = mapped_column(String(255))

    # 一个操作对应一个操作类型
    operation_type: Mapped[OperationType] = relationship(back_populates="operations")

    # 一个操作的手势列表
    gestures: Mapped[List[Gesture]] = relationship(
        secondary=operation_gestures, back_populates="operations"
    )

    def __repr__(self) -> str:
        return f"Operation(id={self.id!r}, type_id={self.type_id!r}), name={self.name!r}), extra_data={self.extra_data!r})"


if __name__ == "__main__":
    from sqlalchemy import create_engine
    engine = create_engine("sqlite:///db/data.db", echo=True)
    from sqlalchemy.orm import Session
    from sqlalchemy import select
    
    session = Session(engine)
    operation = session.scalars(select(Operation).where(Operation.id == 3)).one()
    print(operation)
    
    gestures = operation.gestures
    print(gestures)
    
    