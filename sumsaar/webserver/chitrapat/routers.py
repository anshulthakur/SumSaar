class PipelineRouter:
    """
    A router to control all database operations on models in the
    chitrapat application, directing pipeline/staging models to MongoDB.
    """
    # List of model names that belong to the MongoDB staging area
    mongo_models = {'rawarticle', 'similarityresult', 'pipelinestate'}

    def db_for_read(self, model, **hints):
        if model._meta.model_name in self.mongo_models:
            return 'mongo'
        return 'default'

    def db_for_write(self, model, **hints):
        if model._meta.model_name in self.mongo_models:
            return 'mongo'
        return 'default'

    def allow_relation(self, obj1, obj2, **hints):
        # Allow relations if both are in the same DB
        if obj1._state.db == obj2._state.db:
            return True
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        if model_name in self.mongo_models:
            return db == 'mongo'
        return db == 'default'