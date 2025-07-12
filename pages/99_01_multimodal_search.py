import streamlit as st
from PIL import Image
import io
from cmn.multimodal_search import MultimodalSearchService, config
from typing import Union, Tuple

# Page configuration
st.set_page_config(
    page_title="Multimodal Product Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .status-connected {
        color: #28a745;
        font-weight: bold;
    }
    .status-disconnected {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def validate_image(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded image file"""
    if uploaded_file is None:
        return False, "No file uploaded"

    if uploaded_file.size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"File size exceeds {config.MAX_FILE_SIZE_MB}MB limit"

    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in config.SUPPORTED_IMAGE_FORMATS:
        return False, f"Unsupported format. Use: {', '.join(config.SUPPORTED_IMAGE_FORMATS)}"

    return True, "Valid"

# Header
st.markdown('<h1 class="main-header">🔍 Multimodal Product Search</h1>', unsafe_allow_html=True)

# Initialize service
if 'search_service' not in st.session_state:
    st.session_state.search_service = MultimodalSearchService()

service = st.session_state.search_service

# Sidebar - Connection & Configuration
with st.sidebar:
    st.header("🔌 Connection")

    # Get connection status
    status = service.get_connection_status()

    # Display connection status
    if status['aws_connected']:
        st.markdown('<p class="status-connected">✅ AWS Connected</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-disconnected">❌ AWS Disconnected</p>', unsafe_allow_html=True)

    if status['index_exists']:
        st.markdown('<p class="status-connected">✅ Index Ready</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-disconnected">❌ Index Missing</p>', unsafe_allow_html=True)

    # Connection buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔌 Connect", use_container_width=True):
            with st.spinner("Connecting to AWS..."):
                if service.connect_aws():
                    st.rerun()

    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    st.divider()

    # Index management
    st.header("📊 Index Management")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Create Index", use_container_width=True, disabled=not status['aws_connected']):
            with st.spinner("Creating index..."):
                if service.create_index():
                    st.rerun()

    with col2:
        if st.button("🗑️ Delete Index", use_container_width=True, disabled=not status['aws_connected']):
            if st.session_state.get('confirm_delete', False):
                with st.spinner("Deleting index..."):
                    if service.delete_index():
                        st.session_state.confirm_delete = False
                        st.rerun()
            else:
                st.session_state.confirm_delete = True
                st.warning("Click again to confirm deletion!")
                st.rerun()

    st.divider()

    # Configuration
    st.header("⚙️ Configuration")
    st.info(f"**AWS Profile:** {config.AWS_PROFILE}")
    st.info(f"**Region:** {config.AWS_REGION}")
    st.info(f"**Collection:** {config.COLLECTION_NAME}")
    st.info(f"**Index:** {config.INDEX_NAME}")

    # Statistics
    st.header("📈 Statistics")
    if status['can_operate']:
        stats = service.get_stats()
        if 'error' not in stats:
            st.metric("Total Products", stats['total_products'])
        else:
            st.error(f"Stats error: {stats['error']}")
    else:
        st.info("Connect and create index to see stats")

    # Settings
    st.header("🔧 Settings")
    search_limit = st.slider("Search Results Limit", 1, 50, 10)

# Main content - only show if properly connected
if not status['can_operate']:
    st.warning("⚠️ Please connect to AWS and create an index to use the application.")
    st.info("Use the sidebar controls to:")
    st.info("1. Click '🔌 Connect' to connect to AWS services")
    st.info("2. Click '➕ Create Index' to create the search index")
else:
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📝 Register Product", "🖼️ Search by Image", "📝 Search by Text"])

    # Tab 1: Register Product
    with tab1:
        st.header("📝 Register New Product")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Product Image")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=config.SUPPORTED_IMAGE_FORMATS,
                help=f"Max size: {config.MAX_FILE_SIZE_MB}MB"
            )

            if uploaded_file:
                is_valid, message = validate_image(uploaded_file)
                if is_valid:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Product Image", use_column_width=True)
                else:
                    st.error(f"❌ {message}")
                    image = None
            else:
                image = None

        with col2:
            st.subheader("Product Details")
            title = st.text_input("Product Title", placeholder="Enter product title...")
            description = st.text_area(
                "Product Description", 
                placeholder="Enter detailed product description...",
                height=200
            )

            if st.button("🚀 Register Product", type="primary", use_container_width=True):
                if image and title and description:
                    success = service.register_product(image, title, description)
                    if success:
                        st.balloons()
                else:
                    st.warning("⚠️ Please provide image, title, and description")

    # Tab 2: Search by Image
    with tab2:
        st.header("🖼️ Search by Image")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Upload Search Image")
            search_image_file = st.file_uploader(
                "Choose search image",
                type=config.SUPPORTED_IMAGE_FORMATS,
                key="search_image",
                help="Upload an image to find similar products"
            )

            if search_image_file:
                is_valid, message = validate_image(search_image_file)
                if is_valid:
                    search_image = Image.open(search_image_file)
                    st.image(search_image, caption="Search Image", use_column_width=True)

                    if st.button("🔍 Search Similar Images", type="primary", use_container_width=True):
                        results = service.search_by_image(search_image, search_limit)
                        st.session_state.image_search_results = results
                else:
                    st.error(f"❌ {message}")

        with col2:
            st.subheader("Search Results")
            if 'image_search_results' in st.session_state:
                results = st.session_state.image_search_results
                if results:
                    st.success(f"Found {len(results)} similar products")
                    for i, result in enumerate(results):
                        with st.container():
                            st.markdown(f"""
                            <div class="result-card">
                                <h4>🏷️ {result['title']}</h4>
                                <p><strong>Score:</strong> {result['score']:.4f}</p>
                                <p><strong>Description:</strong> {result['description']}</p>
                                <p><strong>ID:</strong> <code>{result['product_id']}</code></p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No similar products found")

    # Tab 3: Search by Text
    with tab3:
        st.header("📝 Search by Text")

        search_query = st.text_input(
            "Enter search query",
            placeholder="Describe what you're looking for...",
            help="Use natural language to describe the product you want to find"
        )

        col1, col2, col3 = st.columns([1, 1, 2])
        with col2:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)

        if search_button and search_query:
            results = service.search_by_text(search_query, search_limit)

            if results:
                st.success(f"Found {len(results)} matching products")

                # Display results in a nice format
                for i, result in enumerate(results):
                    with st.expander(f"🏷️ {result['title']} (Score: {result['score']:.4f})"):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write(f"**Description:** {result['description']}")
                        with col2:
                            st.code(result['product_id'])
            else:
                st.info("No matching products found. Try different keywords.")
        elif search_button:
            st.warning("⚠️ Please enter a search query")