import pickle
from typing import List, Callable, Dict, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import create_engine, select, update
from sqlalchemy import Select
import numpy as np

from . import schema


def with_commit(session: Session, func: Callable[[Session], None]):
    func(session)
    session.commit()

def no_condition(g: Select[Tuple[schema.Gesture]]) -> Select[Tuple[schema.Gesture]]:
    return g

class DBClient():
    def __init__(self, db_file_path: str, echo=True) -> None:
        cmd = "sqlite:///{}".format(db_file_path)
        engine = create_engine(cmd, echo=echo)
        self.session = Session(engine)

    def add_gesture(self, gesture_name: str):
        def inner(session: Session):
            gesture = schema.Gesture(
                name=gesture_name,
                trained=False
            )
            session.add(gesture)
        with_commit(self.session, inner)

    def add_gesture_data(self, gesture_name: str, dataset: List[List[float]]):
        def inner(session: Session):
            stmt = select(schema.Gesture).where(
                schema.Gesture.name == gesture_name)
            gesture = session.scalars(stmt).one()
            for dataline in dataset:
                gesture.dataset.append(schema.Data(
                    data=pickle.dumps(dataline)
                ))
        with_commit(self.session, inner)

    def add_operation(self, operation_name:str, type_name:str, extra_data:str):
        def inner(session: Session):
            stmt = select(schema.OperationType).where(
                schema.OperationType.type_name == type_name)
            operation_type = session.scalars(stmt).one()
            new_operation = schema.Operation(
                name=operation_name,
                extra_data=extra_data
            )
            operation_type.operations.append(new_operation)
        with_commit(self.session, inner)

    class Dataset():
        def __init__(self, data:List[List[float]], labels: List[int], classes_num: int) -> None:
            self.data = np.array(data)
            self.labels = np.array(labels)
            self.classes_num = classes_num

    def get_dataset(self) -> Dataset:
        datalist = self.session.scalars(select(schema.Data)).all()
        data: List[List[float]] = []
        labels: List[int] = []
        lb_idx = -1
        lb_map:Dict[int, bool] = {}
        for data_record in datalist:
            data.append(pickle.loads(data_record.data))
            if data_record.gesture_id not in lb_map:
                lb_map[data_record.gesture_id] = True
                lb_idx += 1
            labels.append(lb_idx)
        return DBClient.Dataset(data, labels, lb_idx + 1)

    def get_gesture_name_list(self, condition: Callable[[Select[Tuple[schema.Gesture]]], Select[Tuple[schema.Gesture]]] = no_condition) -> List[str]:
        stmt = select(schema.Gesture)
        stmt = condition(stmt)
        gestures = self.session.scalars(stmt).all()
        classes:List[str] = []
        for gesture in gestures:
            classes.append(gesture.name)
        return classes

    def get_operation_gesture_list(self, operation_id: int) -> List[schema.Gesture]:
        stmt = select(schema.Operation).where(
            schema.Operation.id == operation_id)
        operation = self.session.scalars(stmt).one()
        return operation.gestures

    def get_operation_types(self) -> List[schema.OperationType]:
        operation_types = self.session.scalars(
            select(schema.OperationType)
        ).all()
        res = [otype for otype in operation_types]
        return res

    def get_operations(self, type_id=0) -> List[schema.Operation]:
        stmt = select(schema.Operation)
        if type_id != 0:
            stmt = stmt.where(schema.Operation.type_id==type_id)
        operations = self.session.scalars(stmt).all()
        res = [op for op in operations]
        return res

    def get_shape_operation(self, shape_name: str) -> schema.Operation:
        stmt = select(schema.Shape).where(schema.Shape.name == shape_name)
        shape = self.session.scalars(stmt).one()
        return shape.operation

    def update_trained_gestures(self) -> None:
        def inner(session: Session) :
            update_stmt = (
                update(schema.Gesture)
                .values(trained=True)
            )
            session.execute(update_stmt)
        with_commit(self.session, inner)

    def operation_gestures_binding(self, operation_name:str, gesture_names:List[str]) -> None:
        def inner(session: Session) :
            operation = session.scalars(
                select(schema.Operation).
                where(schema.Operation.name == operation_name)
            ).one()
            gestures = session.scalars(
                select(schema.Gesture).
                where(schema.Gesture.name.in_(gesture_names))
            ).all()
            operation.gestures.clear()
            operation.gestures.extend(gestures)
        with_commit(self.session, inner)

    def operation_shape_binding(self, operation_name:str, shape_name:str):
        def inner(session: Session) :
            operation = session.scalars(
                select(schema.Operation).
                where(schema.Operation.name == operation_name)
            ).one()
            shape = session.scalars(
                select(schema.Shape).
                where(schema.Shape.name == shape_name)
            ).one()
            shape.operation = operation
        with_commit(self.session, inner)

    def delete_gesture(self, gesture_id:int):
        def inner(session: Session):
            gesture = session.scalars(
            select(schema.Gesture)\
                .where(schema.Gesture.id == gesture_id)
            ).one()
            session.delete(gesture)
        with_commit(self.session, inner)
