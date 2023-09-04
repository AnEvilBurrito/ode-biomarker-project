import os 

class PathLoader:
    
    def __init__(self, config_path, current_user_path) -> None:
        
        self.config_path = config_path
        self.current_user_path = current_user_path
        
        self.current_user = None
        self.data_path = {}
        self.dir_path = {}
        
        self.get_current_user()
        self._set_config()
        
    def get_data_path(self):
        return self.data_path[self.current_user]
    
    def get_dir_path(self):
        return self.dir_path[self.current_user]
        
    def get_current_user(self):
        
        found = False
        with open(self.current_user_path, 'r') as f:
            data = f.read()
        
        lines = data.split('\n')
        
        for l in lines:
            if l.startswith('CURRENT_USER'):
                self.current_user = l.split('=')[1].strip()
                found = True
                
        if not found:
            raise Exception('CURRENT_USER is not defined in config file')
        
        
    def _set_config(self):
        
        assert self.current_user is not None, 'current user is not defined'
        
        with open(self.config_path, 'r') as f:
            data = f.read()
        
        lines = data.split('\n')
        for l in lines:
            if l.startswith('DATA_PATH'):
                data_path = l.split('=')[1].strip("'")
                user = l.split('=')[0].split('~')[1].strip()
                self.data_path[user] = data_path
            if l.startswith('DIR_PATH'):
                dir_path = l.split('=')[1].strip("'")
                user = l.split('=')[0].split('~')[1].strip()
                self.dir_path[user] = dir_path

        # print(self.data_path)
        # print(self.dir_path)
        # print(self.current_user)

# dl = PathLoader('data_config.env', 'current_user.env')

# print(dl.get_data_path())
# print(dl.get_dir_path())

        