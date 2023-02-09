import pickle
from typing import List, Callable, Dict, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import create_engine, select, update
from sqlalchemy import Select
import numpy as np

from . import schema


class OperationTableRow():
    def __init__(self,
                 operation_id: int,
                 operation_name: str,
                 operation_type: str,
                 extra_data: str,
                 gestures_str: str,
                 shape: str
                 ) -> None:
        self.operation_id = operation_id
        self.operation_name = operation_name
        self.operation_type = operation_type
        self.extra_data = extra_data
        self.gestures_str = gestures_str
        self.shape = shape


class OperationTable():
    def __init__(self) -> None:
        self.header = ['操作名', '操作类型', '参数', '手势列表', '手画图形', '操作']
        self.body: List[OperationTableRow] = []

    def add_row(self, row: OperationTableRow) -> None:
        self.body.append(row)

    def get_body_array(self) -> List[List[str]]:
        res: List[List[str]] = []
        for row in self.body:
            row_list: List[str] = []
            row_list.append(row.operation_name)
            row_list.append(row.operation_type)
            row_list.append(row.extra_data)
            row_list.append(row.gestures_str)
            row_list.append(row.shape)
            row_list.append('')
            res.append(row_list)
        return res

class GestureTableRow():
    def __init__(self, gesture_id: int, gesture_name: str, trained: bool) -> None:
        self.gesture_id = gesture_id
        self.gesture_name = gesture_name
        self.trained = trained

class GestureTable():
    def __init__(self) -> None:
        self.header = ['手势', '状态', '操作']
        self.body: List[GestureTableRow] = []

    def add_row(self, row: GestureTableRow) -> None:
        self.body.append(row)

    def get_body_array(self) -> List[List[str]]:
        res: List[List[str]] = []
        for row in self.body:
            row_list: List[str] = []
            row_list.append(row.gesture_name)
            row_list.append("已训练" if row.trained else "未训练")
            row_list.append('')
            res.append(row_list)
        return res


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

    def add_operation(self, operation_name: str, type_name: str, extra_data: str):
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
        def __init__(self, data: List[List[float]], labels: List[int], classes_num: int) -> None:
            self.data = np.array(data)
            self.labels = np.array(labels)
            self.classes_num = classes_num

    def get_dataset(self) -> Dataset:
        datalist = self.session.scalars(select(schema.Data)).all()
        data: List[List[float]] = []
        labels: List[int] = []
        lb_idx = -1
        lb_map: Dict[int, bool] = {}
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
        classes: List[str] = []
        for gesture in gestures:
            classes.append(gesture.name)
        return classes

    def get_gesture(self, gesture_id: int) -> schema.Gesture:
        stmt = select(schema.Gesture).where(
            schema.Gesture.id == gesture_id)
        gesture = self.session.scalars(stmt).one()
        return gesture

    def get_operation(self, operation_id: int) -> schema.Operation:
        stmt = select(schema.Operation).where(
            schema.Operation.id == operation_id)
        operation = self.session.scalars(stmt).one()
        return operation

    def get_operation_types(self) -> List[schema.OperationType]:
        operation_types = self.session.scalars(
            select(schema.OperationType)
        ).all()
        res = [otype for otype in operation_types]
        return res

    def get_operations(self, type_id=0) -> List[schema.Operation]:
        stmt = select(schema.Operation)
        if type_id != 0:
            stmt = stmt.where(schema.Operation.type_id == type_id)
        operations = self.session.scalars(stmt).all()
        res = [op for op in operations]
        return res

    def get_shape_operation(self, shape_name: str):
        stmt = select(schema.Shape).where(schema.Shape.name == shape_name)
        shape = self.session.scalars(stmt).one()
        return shape.operation

    def get_gestures_operation_mapping(self) -> Dict[str, schema.Operation]:
        mapping: Dict[str, schema.Operation] = {}
        operations = self.session.scalars(select(schema.Operation)).all()
        for operation in operations:
            gestures = operation.gestures
            gesture_name_list_str = "+".join(
                list(map(lambda g: g.name, gestures)))
            if gesture_name_list_str == "":
                continue
            mapping[gesture_name_list_str] = operation
        return mapping

    def get_operation_table(self) -> OperationTable:
        table = OperationTable()
        operations = self.get_operations()
        for op in operations:
            gestures_str = "+".join(list(map(lambda g: g.name, op.gestures)))
            shape = op.shape.name if op.shape != None else ""
            print(op.name, gestures_str, shape)
            table.add_row(OperationTableRow(
                op.id, op.name, op.operation_type.type_name, 
                op.extra_data, gestures_str, shape))
        return table

    def get_gesture_table(self) -> GestureTable:
        table = GestureTable()
        gestures = self.session.scalars(select(schema.Gesture)).all()
        for gesture in gestures:
            row = GestureTableRow(gesture.id, gesture.name, gesture.trained)
            table.add_row(row)
        return table

    def update_trained_gestures(self) -> None:
        def inner(session: Session):
            update_stmt = (
                update(schema.Gesture)
                .values(trained=True)
            )
            session.execute(update_stmt)
        with_commit(self.session, inner)

    def update_operation(self, operation_id: int, operation_name: str, extra_data: str):
        def inner(session: Session):
            update_stmt = (
                update(schema.Operation)
                .where(schema.Operation.id == operation_id)
                .values(name=operation_name)
                .values(extra_data=extra_data)
            )
            session.execute(update_stmt)
        with_commit(self.session, inner)

    def operation_gestures_binding(self, operation_id: int, gesture_names: List[str]) -> None:
        def inner(session: Session):
            operation = session.scalars(
                select(schema.Operation).
                where(schema.Operation.id == operation_id)
            ).one()
            gestures = session.scalars(
                select(schema.Gesture).
                where(schema.Gesture.name.in_(gesture_names))
            ).all()
            operation.gestures.clear()
            operation.gestures.extend(gestures)
        with_commit(self.session, inner)

    def operation_shape_binding(self, operation_id: int, shape_name: str):
        def inner(session: Session):
            operation = session.scalars(
                select(schema.Operation).
                where(schema.Operation.id == operation_id)
            ).one()
            shape = session.scalars(
                select(schema.Shape).
                where(schema.Shape.name == shape_name)
            ).one_or_none()
            operation.shape = shape
        with_commit(self.session, inner)

    def delete_gesture(self, gesture_id: int):
        def inner(session: Session):
            gesture = session.scalars(
                select(schema.Gesture)
                .where(schema.Gesture.id == gesture_id)
            ).one()
            session.delete(gesture)
        with_commit(self.session, inner)

    def delete_operation(self, operation_id: int):
        def inner(session: Session):
            operation = session.scalars(
                select(schema.Operation)
                .where(schema.Operation.id == operation_id)
            ).one()
            session.delete(operation)
        with_commit(self.session, inner)
