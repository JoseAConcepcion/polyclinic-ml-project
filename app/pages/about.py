import streamlit as st
from modules.nav import Navbar

def main():
    Navbar()
    st.header("Acerca de nosotros")
    st.write("Link al repo al informe aqui atec etc")
    
if __name__ == '__main__':
    main()